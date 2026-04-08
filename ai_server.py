# -*- coding: utf-8 -*-
"""
번아웃 감지 AI 서버 v2.13
=======================

POST /analyze                  : 분석 요청 -> 즉시 200 OK -> 큐 추가 -> 백그라운드 분석 -> 콜백
POST /analysis/cancel/{diary_id}: 분석 취소 (큐 제거 + 플래그 등록)
POST /feedback/batch           : 백엔드 2주 배치 피드백 수신 및 CSV 저장
GET  /feedback/stats           : 누적 피드백 통계 조회

실행: uvicorn ai_server:app --reload --port 8001
"""

import os
import json
import httpx
import random
import asyncio
from collections import deque
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 분할된 모듈 임포트
from config import Config
from constants import (
    PERSONA_MAP,
    BURNOUT_TO_ACTIVITY_CATEGORY,
    ACTIVITY_CATEGORY_IDS,
    ACTIVITY_CONTENT,
    ACTIVITY_ATTRIBUTES,
    PREFERENCE_SIGNAL_KEYWORDS,
)
from models import (
    AnalyzeRequest,
    AnalysisCallback,
    DiaryHistory,
    DiaryAnalysisResult,
    RecommendationItem,
    FeedbackBatchRequest,
    FeedbackBatchResponse,
)
from feedback_store import FeedbackStore, VALID_MBI_CATEGORIES
from analyzer import BurnoutAnalyzer
from feedback import FeedbackGenerator
from emotion_match import EmotionMatchChecker
from insight import StatisticsInsightGenerator
from prompts import PersonaType
from error_codes import (
    ErrorCode,
    ErrorDetail,
    AIServerException,
    create_error,
    get_fallback_feedback,
    ERROR_DEFINITIONS,
)


# ============================================
# 글로벌 인스턴스
# ============================================

analyzer: Optional[BurnoutAnalyzer] = None
feedback_gen: Optional[FeedbackGenerator] = None
emotion_checker: Optional[EmotionMatchChecker] = None
insight_gen: Optional[StatisticsInsightGenerator] = None
feedback_store: Optional[FeedbackStore] = None

# 분석 취소 요청된 diary_id set
cancelled_diary_ids: set = set()

# 분석 큐 (deque 기반 단일 worker)
analysis_queue: deque = deque()
analysis_queue_lock = asyncio.Lock()
analysis_worker_task = None


# ============================================
# FastAPI 앱
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer, feedback_gen, emotion_checker, insight_gen, feedback_store, analysis_worker_task

    try:
        analyzer = BurnoutAnalyzer()
        analyzer.initialize()
    except RuntimeError as e:
        print(f"[경고] 모델 로드 실패: {e}")
        print("[경고] 서버는 실행되지만 분석 요청은 실패합니다.")
        analyzer = BurnoutAnalyzer()  # _initialized=False 상태 유지

    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    print(f"피드백 모드: {'LLM (KoAlpaca)' if use_llm else '템플릿'}")

    feedback_gen = FeedbackGenerator(use_llm=use_llm)
    emotion_checker = EmotionMatchChecker()
    insight_gen = StatisticsInsightGenerator()
    feedback_store = FeedbackStore()

    analysis_worker_task = asyncio.create_task(analysis_worker())

    yield

    analysis_worker_task.cancel()
    try:
        await analysis_worker_task
    except asyncio.CancelledError:
        pass
    print("서버 종료")


app = FastAPI(
    title="번아웃 감지 AI 서버",
    description="한국형 번아웃 감정 분석 API",
    version="2.13.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================
# API 엔드포인트
# ============================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Burnout Detection AI Server",
        "version": "2.13.0",
        "device": Config.DEVICE,
        "model_loaded": analyzer is not None and analyzer._initialized,
        "features": [
            "emotion_analysis",
            "emotion_match_check",
            "statistics_insight",
            "activity_recommendation",
            "user_feedback",
        ],
    }


@app.post("/feedback/batch", response_model=FeedbackBatchResponse)
async def receive_feedback_batch(request: FeedbackBatchRequest):
    """
    배치 피드백 수신

    백엔드가 누적된 설문 응답을 일괄 전송합니다.
    - feedbacks : 피드백 레코드 목록
      - predicted_mbi_category : AI가 예측한 카테고리
      - ai_message_rating      : AI 메시지 만족도 1~5
      - mbi_category_rating    : MBI 카테고리 만족도 1~5
    """
    if not request.feedbacks:
        raise HTTPException(status_code=400, detail="feedbacks가 비어있습니다.")

    for i, rec in enumerate(request.feedbacks):
        if not (1 <= rec.ai_message_rating <= 5):
            raise HTTPException(status_code=400, detail=f"feedbacks[{i}].ai_message_rating는 1~5 사이여야 합니다.")
        if not (1 <= rec.mbi_category_rating <= 5):
            raise HTTPException(status_code=400, detail=f"feedbacks[{i}].mbi_category_rating는 1~5 사이여야 합니다.")
        if rec.predicted_mbi_category not in VALID_MBI_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"feedbacks[{i}].predicted_mbi_category 유효하지 않음: {rec.predicted_mbi_category}")

    result = feedback_store.save_batch(records=[rec.model_dump() for rec in request.feedbacks])

    return FeedbackBatchResponse(
        status="saved",
        received=result["received"],
        total_accumulated=result["total_accumulated"],
        avg_ai_message_rating=result["avg_ai_message_rating"],
        avg_mbi_category_rating=result["avg_mbi_category_rating"],
        low_mbi_by_category=result["low_mbi_by_category"],
    )


@app.get("/feedback/stats")
async def get_feedback_stats():
    """누적 피드백 통계 조회"""
    return feedback_store.get_stats()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_diary(request: AnalyzeRequest):
    """일기 분석 요청 (비동기)"""
    if not request.history:
        raise HTTPException(status_code=400, detail="history가 비어있습니다.")

    async with analysis_queue_lock:
        cancelled_diary_ids.discard(request.diary_id)
        analysis_queue.append((request.diary_id, request.user_id, request.persona, request.history))

    return {"status": "accepted", "message": "분석이 시작되었습니다."}


@app.post("/analyze/sync")
async def analyze_sync(request: AnalyzeRequest):
    """동기 분석 (테스트용) — analyze_batch 사용"""
    if not request.history:
        raise HTTPException(status_code=400, detail="history가 비어있습니다.")

    persona_type = PERSONA_MAP.get(request.persona, PersonaType.WARM_COUNSELOR)
    feedback_gen.set_persona(persona_type)
    emotion_checker.set_persona(persona_type)
    insight_gen.set_persona(persona_type)

    batch_items = [{"text": d.content or "", "keywords": d.keywords or {}} for d in request.history]
    batch_results = analyzer.analyze_batch(batch_items)

    diary_analyses = [
        DiaryAnalysisResult(
            diary_id=d.diary_id,
            primary_emotion=r["primary_emotion"],
            primary_score=round(r["primary_score"], 4),
            mbi_category=r["mbi_category"],
            keywords=r.get("keywords", [])
        )
        for d, r in zip(request.history, batch_results)
    ]

    today_diary = request.history[0]
    today_result = batch_results[0]
    category = "긍정" if today_result["primary_emotion"] == "긍정" else today_result.get("burnout_category", "정서적_고갈")

    emotion_match = None
    if today_diary.keywords:
        emotion_match = emotion_checker.check_match(today_diary.keywords, today_result)

    statistics_insight = insight_gen.generate(diary_analyses, request.history)

    ai_message = feedback_gen.generate(
        category=category,
        user_text=today_diary.content or "",
        keywords=today_result.get("keywords", [])
    )
    if emotion_match and not emotion_match.is_matched and emotion_match.hidden_emotion_hint:
        ai_message = f"{ai_message}\n\n{emotion_match.hidden_emotion_hint}"

    recommendations = []
    if len(request.history) >= Config.MIN_DIARY_COUNT_FOR_RECOMMENDATION:
        recommendations = generate_recommendations(category, today_diary.content or "", today_result.get("keywords", []))

    return AnalysisCallback(
        diary_id=request.diary_id,
        primary_emotion=today_result["primary_emotion"],
        primary_score=round(today_result["primary_score"], 4),
        mbi_category=today_result["mbi_category"],
        emotion_probs=today_result["emotion_probs"],
        ai_message=ai_message,
        diary_analyses=diary_analyses,
        recommendations=recommendations,
        emotion_match=emotion_match,
        statistics_insight=statistics_insight
    )


@app.get("/personas")
async def list_all_personas():
    """페르소나 목록"""
    return {
        "personas": [
            {"type": "warm_counselor", "name": "따뜻한 상담사", "tone": "부드럽고 다정한"},
            {"type": "practical_advisor", "name": "실용적 조언자", "tone": "차분하고 명확한"},
            {"type": "friendly_buddy", "name": "친근한 친구", "tone": "편하고 친근한"},
            {"type": "calm_mentor", "name": "차분한 멘토", "tone": "담담하고 깊이 있는"},
            {"type": "cheerful_supporter", "name": "밝은 응원단", "tone": "밝고 에너지 넘치는"},
        ]
    }


@app.post("/config/activities")
async def set_activity_ids(activities: Dict[str, List[int]]):
    """활동 ID 설정"""
    for category, ids in activities.items():
        if category in ACTIVITY_CATEGORY_IDS:
            ACTIVITY_CATEGORY_IDS[category] = ids
    return {"status": "updated", "activities": ACTIVITY_CATEGORY_IDS}


@app.get("/config/activities")
async def get_activity_ids():
    """현재 활동 ID 조회"""
    return {"activities": ACTIVITY_CATEGORY_IDS}


@app.post("/analysis/cancel/{diary_id}")
async def cancel_analysis(diary_id: int):
    """
    일기 분석 취소 요청.
    큐에서 즉시 제거 + 실행 중이면 다음 체크포인트에서 중단.
    """
    async with analysis_queue_lock:
        cancelled_diary_ids.add(diary_id)
        to_keep = [item for item in analysis_queue if item[0] != diary_id]
        analysis_queue.clear()
        analysis_queue.extend(to_keep)
    print(f"분석 취소 요청: diary_id={diary_id}")
    return {"message": "Analysis cancelled or ignored"}


@app.get("/errors")
async def list_error_codes():
    """에러 코드 목록 조회"""
    error_list = []
    for code in ErrorCode:
        definition = ERROR_DEFINITIONS.get(code, {})
        error_list.append({
            "code": code.value,
            "name": code.name,
            "message": definition.get("message", ""),
            "recoverable": definition.get("recoverable", False)
        })
    return {
        "total": len(error_list),
        "categories": {
            "AI1xxx": "모델 관련 에러",
            "AI2xxx": "입력 데이터 관련 에러",
            "AI3xxx": "분석 처리 관련 에러",
            "AI4xxx": "외부 통신 관련 에러",
            "AI5xxx": "시스템 관련 에러"
        },
        "errors": error_list
    }


# ============================================
# 분석 큐 worker
# ============================================

async def analysis_worker():
    """단일 worker — 큐에서 순서대로 꺼내 처리 (GPU 동시 접근 없음)"""
    while True:
        async with analysis_queue_lock:
            item = analysis_queue.popleft() if analysis_queue else None
        if item:
            await process_analysis(*item)
        else:
            await asyncio.sleep(0.1)


# ============================================
# 백그라운드 처리 함수
# ============================================

async def process_analysis(diary_id: int, user_id: int, persona, history: List[DiaryHistory]):
    """백그라운드 분석 — 히스토리 배치 처리로 임베딩 1회 수행"""
    errors: List[ErrorDetail] = []
    fallback_used = False

    try:
        print(f"분석 시작: diary_id={diary_id}, user_id={user_id}, persona={persona}")

        # 취소 여부 확인 (시작 전)
        if diary_id in cancelled_diary_ids:
            cancelled_diary_ids.discard(diary_id)
            print(f"분석 취소됨 (시작 전): diary_id={diary_id}")
            return

        # 모델 로드 확인
        if not analyzer or not analyzer._initialized:
            raise AIServerException(ErrorCode.MODEL_NOT_LOADED)

        # 페르소나 설정
        persona_type = PERSONA_MAP.get(persona)
        if persona_type is None:
            errors.append(create_error(ErrorCode.INVALID_PERSONA, f"'{persona}' -> 기본값 사용"))
            persona_type = PersonaType.WARM_COUNSELOR

        feedback_gen.set_persona(persona_type)
        emotion_checker.set_persona(persona_type)
        insight_gen.set_persona(persona_type)

        # ── 히스토리 전체 배치 분석 (임베딩 1회) ──
        today_diary = history[0]
        diary_analyses = []
        today_result = None
        category = "긍정"

        if today_diary.content and len((today_diary.content or "").strip()) < 10:
            errors.append(create_error(ErrorCode.CONTENT_TOO_SHORT, f"내용 길이: {len(today_diary.content)}자"))

        try:
            batch_items = [
                {"text": d.content or "", "keywords": d.keywords or {}}
                for d in history
            ]
            batch_results = analyzer.analyze_batch(batch_items)

            for diary, result in zip(history, batch_results):
                diary_analyses.append(DiaryAnalysisResult(
                    diary_id=diary.diary_id,
                    primary_emotion=result["primary_emotion"],
                    primary_score=round(result["primary_score"], 4),
                    mbi_category=result["mbi_category"],
                    keywords=result.get("keywords", [])
                ))

            today_result = batch_results[0]
            category = "긍정" if today_result["primary_emotion"] == "긍정" else today_result.get("burnout_category", "정서적_고갈")

        except Exception as e:
            errors.append(create_error(ErrorCode.ANALYSIS_FAILED, str(e)))
            # 배치 실패 시 단건 fallback
            for diary in history:
                try:
                    result = analyzer.analyze(diary.content or "", diary.keywords or {})
                    diary_analyses.append(DiaryAnalysisResult(
                        diary_id=diary.diary_id,
                        primary_emotion=result["primary_emotion"],
                        primary_score=round(result["primary_score"], 4),
                        mbi_category=result["mbi_category"],
                        keywords=result.get("keywords", [])
                    ))
                except Exception as e2:
                    errors.append(create_error(ErrorCode.ANALYSIS_FAILED, f"diary_id={diary.diary_id}: {str(e2)}"))
                    diary_analyses.append(DiaryAnalysisResult(
                        diary_id=diary.diary_id,
                        primary_emotion="긍정",
                        primary_score=0.5,
                        mbi_category="NORMAL",
                        keywords=[]
                    ))
            today_result = {
                "primary_emotion": "긍정", "primary_score": 0.5, "mbi_category": "NORMAL",
                "emotion_probs": {"긍정": 0.5, "부정": 0.5}, "keywords": []
            }
            fallback_used = True

        if today_result is None:
            today_result = {
                "primary_emotion": "긍정", "primary_score": 0.5, "mbi_category": "NORMAL",
                "emotion_probs": {"긍정": 0.5, "부정": 0.5}, "keywords": []
            }
            fallback_used = True

        # 취소 여부 확인 (분석 후)
        if diary_id in cancelled_diary_ids:
            cancelled_diary_ids.discard(diary_id)
            print(f"분석 취소됨 (분석 후): diary_id={diary_id}")
            return

        # 감정 일치도 검사
        emotion_match = None
        if today_diary.keywords:
            try:
                emotion_match = emotion_checker.check_match(today_diary.keywords, today_result)
            except Exception as e:
                errors.append(create_error(ErrorCode.EMOTION_MATCH_FAILED, str(e)))

        # 통계 인사이트 생성
        statistics_insight = None
        try:
            statistics_insight = insight_gen.generate(diary_analyses, history)
        except Exception as e:
            errors.append(create_error(ErrorCode.INSIGHT_GENERATION_FAILED, str(e)))

        # 피드백 생성
        ai_message = ""
        try:
            ai_message = feedback_gen.generate(
                category=category,
                user_text=today_diary.content or "",
                keywords=today_result.get("keywords", [])
            )
            if emotion_match and not emotion_match.is_matched and emotion_match.hidden_emotion_hint:
                ai_message = f"{ai_message}\n\n{emotion_match.hidden_emotion_hint}"
        except Exception as e:
            errors.append(create_error(ErrorCode.FEEDBACK_GENERATION_FAILED, str(e), fallback_used=True))
            ai_message = get_fallback_feedback(category)
            fallback_used = True

        # 취소 여부 확인 (피드백 생성 후)
        if diary_id in cancelled_diary_ids:
            cancelled_diary_ids.discard(diary_id)
            print(f"분석 취소됨 (피드백 후): diary_id={diary_id}")
            return

        # 활동 추천
        recommendations = []
        if len(history) >= Config.MIN_DIARY_COUNT_FOR_RECOMMENDATION:
            try:
                recommendations = generate_recommendations(category, today_diary.content or "", today_result.get("keywords", []))
            except Exception as e:
                errors.append(create_error(ErrorCode.RECOMMENDATION_FAILED, str(e)))

        # 콜백 전송
        callback_data = AnalysisCallback(
            diary_id=diary_id,
            primary_emotion=today_result["primary_emotion"],
            primary_score=round(today_result["primary_score"], 4),
            mbi_category=today_result["mbi_category"],
            emotion_probs=today_result["emotion_probs"],
            ai_message=ai_message,
            diary_analyses=diary_analyses,
            recommendations=recommendations,
            emotion_match=emotion_match,
            statistics_insight=statistics_insight,
            success=True,
            errors=errors,
            fallback_used=fallback_used
        )

        await send_callback(callback_data)
        print(f"분석 완료: diary_id={diary_id}, 일기수={len(diary_analyses)}, 추천수={len(recommendations)}, 에러={len(errors)}건, 폴백={fallback_used}")

    except AIServerException as e:
        print(f"분석 실패 (AI에러): diary_id={diary_id}, code={e.code}, message={e.message}")
        errors.append(e.error)
        fail_callback = AnalysisCallback(
            diary_id=diary_id, primary_emotion="긍정", primary_score=0.5,
            mbi_category="NORMAL", emotion_probs={"긍정": 0.5, "부정": 0.5},
            ai_message=get_fallback_feedback("default"), diary_analyses=[],
            recommendations=[], success=False, errors=errors, fallback_used=True
        )
        await send_callback(fail_callback)

    except Exception as e:
        print(f"분석 실패 (내부오류): diary_id={diary_id}, error={e}")
        import traceback
        traceback.print_exc()
        errors.append(create_error(ErrorCode.INTERNAL_ERROR, str(e)))
        fail_callback = AnalysisCallback(
            diary_id=diary_id, primary_emotion="긍정", primary_score=0.5,
            mbi_category="NORMAL", emotion_probs={"긍정": 0.5, "부정": 0.5},
            ai_message=get_fallback_feedback("default"), diary_analyses=[],
            recommendations=[], success=False, errors=errors, fallback_used=True
        )
        await send_callback(fail_callback)


# ============================================
# 헬퍼 함수
# ============================================

def infer_user_preference(user_text: str, keywords: List[str]) -> Dict[str, Optional[bool]]:
    """user_text + keywords에서 성향(is_active/is_outdoor/is_social) 추론"""
    combined = user_text + " " + " ".join(keywords)
    result: Dict[str, Optional[bool]] = {"is_active": None, "is_outdoor": None, "is_social": None}
    for attr, signals in PREFERENCE_SIGNAL_KEYWORDS.items():
        if any(kw in combined for kw in signals["avoid"]):
            result[attr] = False
        elif any(kw in combined for kw in signals["prefer"]):
            result[attr] = True
    return result


def matches_preference(attrs: Dict, preference: Dict[str, Optional[bool]]) -> bool:
    """비선호 조건에 걸리지 않으면 True"""
    for attr, pref_val in preference.items():
        if pref_val is False and attrs.get(attr, False):
            return False
    return True


def generate_recommendations(category: str, user_text: str, keywords: List[str]) -> List[RecommendationItem]:
    """활동 추천 생성 (성향 필터링 + LLM ai_message 선택적 생성)"""
    recommendations = []
    if category == "긍정" or category is None:
        return recommendations

    preference = infer_user_preference(user_text, keywords)
    activity_categories = BURNOUT_TO_ACTIVITY_CATEGORY.get(category, ["REST", "SMALL_WIN"])

    for act_cat in activity_categories:
        activity_ids = ACTIVITY_CATEGORY_IDS.get(act_cat, [])
        if not activity_ids:
            continue

        candidates = [
            act_id for act_id in activity_ids
            if matches_preference(ACTIVITY_ATTRIBUTES.get(act_id, {}), preference)
        ]
        selected_id = random.choice(candidates) if candidates else random.choice(activity_ids)
        act_content = ACTIVITY_CONTENT.get(selected_id, "")
        attrs = ACTIVITY_ATTRIBUTES.get(selected_id, {
            "act_category": act_cat, "is_active": False, "is_outdoor": False, "is_social": False,
        })

        ai_message = ""
        if feedback_gen and feedback_gen.use_llm:
            try:
                ai_message = feedback_gen.generate(
                    category=category, user_text=user_text,
                    keywords=keywords, activity_name=act_content,
                )
            except Exception:
                ai_message = ""

        recommendations.append(RecommendationItem(
            act_content=act_content,
            act_category=attrs["act_category"],
            is_active=attrs["is_active"],
            is_outdoor=attrs["is_outdoor"],
            is_social=attrs["is_social"],
            ai_message=ai_message,
        ))

    return recommendations


async def send_callback(data: AnalysisCallback):
    """백엔드 콜백 전송"""
    try:
        emotion_probs: dict = dict(data.emotion_probs)
        if data.statistics_insight:
            si = data.statistics_insight
            total = si.total_entries
            burnout_trend = si.burnout_trend
            emotion_probs["statistics"] = {
                "period": si.period,
                "total_entries": total,
                "emotion_frequency": si.emotion_frequency,
                "burnout_trend": burnout_trend,
                "mbi_distribution": {cat: round(cnt / total, 4) for cat, cnt in burnout_trend.items()},
                "situation_frequency": si.situation_frequency,
                "top_keywords": si.top_keywords,
                "insight_messages": si.insight_messages,
            }

        payload = {
            "diary_id": data.diary_id,
            "primary_emotion": data.primary_emotion,
            "primary_score": data.primary_score,
            "mbi_category": data.mbi_category,
            "emotion_probs": emotion_probs,
            "ai_message": data.ai_message,
            "recommendations": [
                {
                    "act_content": r.act_content,
                    "act_category": r.act_category,
                    "is_active": r.is_active,
                    "is_outdoor": r.is_outdoor,
                    "is_social": r.is_social,
                    "ai_message": r.ai_message,
                }
                for r in data.recommendations
            ],
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                Config.BACKEND_CALLBACK_URL,
                content=json.dumps(payload, ensure_ascii=False),
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                print(f"콜백 성공: diary_id={data.diary_id}")
            else:
                print(f"콜백 실패: diary_id={data.diary_id}, status={response.status_code}, body={response.text}")
    except httpx.TimeoutException:
        print(f"콜백 타임아웃: diary_id={data.diary_id}, url={Config.BACKEND_CALLBACK_URL}")
    except httpx.ConnectError:
        print(f"콜백 연결 실패: diary_id={data.diary_id}, url={Config.BACKEND_CALLBACK_URL} (백엔드 서버 확인 필요)")
    except Exception as e:
        print(f"콜백 에러: diary_id={data.diary_id}, error={type(e).__name__}: {e}")


# ============================================
# 실행
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8001, reload=True)
