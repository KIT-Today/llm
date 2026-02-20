# -*- coding: utf-8 -*-
"""
번아웃 감지 AI 서버 v2.3
=======================

POST /analyze : 분석 요청 -> 즉시 200 OK -> 백그라운드 분석 -> 콜백

실행: uvicorn ai_server:app --reload --port 8001
"""

import os
import httpx
import random
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 분할된 모듈 임포트
from config import Config
from constants import (
    PERSONA_MAP,
    BURNOUT_TO_ACTIVITY_CATEGORY,
    ACTIVITY_CATEGORY_IDS,
    ACTIVITY_CONTENT,
)
from models import (
    AnalyzeRequest,
    AnalysisCallback,
    DiaryHistory,
    DiaryAnalysisResult,
    RecommendationItem,
)
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


# ============================================
# FastAPI 앱
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer, feedback_gen, emotion_checker, insight_gen
    
    analyzer = BurnoutAnalyzer()
    analyzer.initialize()
    
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    print(f"피드백 모드: {'LLM (KoAlpaca)' if use_llm else '템플릿'}")
    
    feedback_gen = FeedbackGenerator(use_llm=use_llm)
    emotion_checker = EmotionMatchChecker()
    insight_gen = StatisticsInsightGenerator()
    
    yield
    print("서버 종료")


app = FastAPI(
    title="번아웃 감지 AI 서버",
    description="한국형 번아웃 감정 분석 API",
    version="2.3.0",
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
        "version": "2.3.0",
        "device": Config.DEVICE,
        "model_loaded": analyzer is not None and analyzer._initialized,
        "features": ["emotion_analysis", "emotion_match_check", "statistics_insight", "activity_recommendation"]
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_diary(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """일기 분석 요청 (비동기)"""
    if not request.history:
        raise HTTPException(status_code=400, detail="history가 비어있습니다.")
    
    background_tasks.add_task(
        process_analysis,
        diary_id=request.diary_id,
        user_id=request.user_id,
        persona=request.persona,
        history=request.history
    )
    
    return {"status": "accepted", "message": "분석이 시작되었습니다."}


@app.post("/analyze/sync")
async def analyze_sync(request: AnalyzeRequest):
    """동기 분석 (테스트용)"""
    if not request.history:
        raise HTTPException(status_code=400, detail="history가 비어있습니다.")
    
    persona_type = PERSONA_MAP.get(request.persona, PersonaType.WARM_COUNSELOR)
    feedback_gen.set_persona(persona_type)
    emotion_checker.set_persona(persona_type)
    insight_gen.set_persona(persona_type)
    
    # 모든 일기 분석
    diary_analyses = []
    for diary in request.history:
        result = analyzer.analyze(diary.content or "", diary.keywords or {})
        diary_analyses.append(DiaryAnalysisResult(
            diary_id=diary.diary_id,
            primary_emotion=result["primary_emotion"],
            primary_score=round(result["primary_score"], 4),
            mbi_category=result["mbi_category"],
            keywords=result.get("keywords", [])
        ))
    
    # 오늘 일기 분석
    today_diary = request.history[0]
    today_result = analyzer.analyze(today_diary.content or "", today_diary.keywords or {})
    category = "긍정" if today_result["primary_emotion"] == "긍정" else today_result.get("burnout_category", "정서적_고갈")
    
    # 감정 일치도 검사
    emotion_match = None
    if today_diary.keywords:
        emotion_match = emotion_checker.check_match(today_diary.keywords, today_result)
    
    # 통계 인사이트
    statistics_insight = insight_gen.generate(diary_analyses, request.history)
    
    # 피드백 생성
    ai_message = feedback_gen.generate(
        category=category,
        user_text=today_diary.content or "",
        keywords=today_result.get("keywords", [])
    )
    
    if emotion_match and not emotion_match.is_matched and emotion_match.hidden_emotion_hint:
        ai_message = f"{ai_message}\n\n{emotion_match.hidden_emotion_hint}"
    
    # 활동 추천
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
# 백그라운드 처리 함수
# ============================================

async def process_analysis(diary_id: int, user_id: int, persona, history: List[DiaryHistory]):
    """백그라운드 분석"""
    errors: List[ErrorDetail] = []
    fallback_used = False
    
    try:
        print(f"분석 시작: diary_id={diary_id}, user_id={user_id}, persona={persona}")
        
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
        
        # 모든 일기 분석
        diary_analyses = []
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
            except Exception as e:
                errors.append(create_error(ErrorCode.ANALYSIS_FAILED, f"diary_id={diary.diary_id}: {str(e)}"))
                diary_analyses.append(DiaryAnalysisResult(
                    diary_id=diary.diary_id,
                    primary_emotion="긍정",
                    primary_score=0.5,
                    mbi_category="NONE",
                    keywords=[]
                ))
                fallback_used = True
        
        # 오늘 일기 분석
        today_diary = history[0]
        today_result = None
        category = "긍정"
        
        try:
            if today_diary.content and len(today_diary.content.strip()) < 10:
                errors.append(create_error(ErrorCode.CONTENT_TOO_SHORT, f"내용 길이: {len(today_diary.content)}자"))
            
            today_result = analyzer.analyze(today_diary.content or "", today_diary.keywords or {})
            category = "긍정" if today_result["primary_emotion"] == "긍정" else today_result.get("burnout_category", "정서적_고갈")
        except Exception as e:
            errors.append(create_error(ErrorCode.STAGE1_INFERENCE_FAILED, str(e)))
            today_result = {
                "primary_emotion": "긍정", "primary_score": 0.5, "mbi_category": "NONE",
                "emotion_probs": {"긍정": 0.5, "부정": 0.5}, "keywords": []
            }
            fallback_used = True
        
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
        
        print(f"분석 완료: diary_id={diary_id}, 일기수={len(diary_analyses)}, 추천수={len(recommendations)}, "
              f"에러={len(errors)}건, 폴백={fallback_used}")
        
    except AIServerException as e:
        print(f"분석 실패 (AI에러): diary_id={diary_id}, code={e.code}, message={e.message}")
        errors.append(e.error)
        
        fail_callback = AnalysisCallback(
            diary_id=diary_id,
            primary_emotion="긍정",
            primary_score=0.5,
            mbi_category="NONE",
            emotion_probs={"긍정": 0.5, "부정": 0.5},
            ai_message=get_fallback_feedback("default"),
            diary_analyses=[],
            recommendations=[],
            success=False,
            errors=errors,
            fallback_used=True
        )
        await send_callback(fail_callback)
        
    except Exception as e:
        print(f"분석 실패 (내부오류): diary_id={diary_id}, error={e}")
        import traceback
        traceback.print_exc()
        
        errors.append(create_error(ErrorCode.INTERNAL_ERROR, str(e)))
        
        fail_callback = AnalysisCallback(
            diary_id=diary_id,
            primary_emotion="긍정",
            primary_score=0.5,
            mbi_category="NONE",
            emotion_probs={"긍정": 0.5, "부정": 0.5},
            ai_message=get_fallback_feedback("default"),
            diary_analyses=[],
            recommendations=[],
            success=False,
            errors=errors,
            fallback_used=True
        )
        await send_callback(fail_callback)


def generate_recommendations(category: str, user_text: str, keywords: List[str]) -> List[RecommendationItem]:
    """활동 추천 생성"""
    recommendations = []
    
    if category == "긍정" or category is None:
        return recommendations
    
    activity_categories = BURNOUT_TO_ACTIVITY_CATEGORY.get(category, ["REST", "SMALL_WIN"])
    
    for act_category in activity_categories:
        activity_ids = ACTIVITY_CATEGORY_IDS.get(act_category, [])
        if activity_ids:
            selected_id = random.choice(activity_ids)
            activity_name = ACTIVITY_CONTENT.get(selected_id, "")
            
            ai_message = feedback_gen.generate(
                category=category,
                user_text=user_text,
                keywords=keywords,
                activity_name=activity_name
            )
            recommendations.append(RecommendationItem(activity_id=selected_id, ai_message=ai_message))
    
    return recommendations


async def send_callback(data: AnalysisCallback):
    """백엔드 콜백 전송"""
    try:
        # 백엔드 AIAnalysisResult 스키마에 맞는 필드만 전송
        payload = {
            "diary_id": data.diary_id,
            "primary_emotion": data.primary_emotion,
            "primary_score": data.primary_score,
            "mbi_category": data.mbi_category,
            "emotion_probs": data.emotion_probs,
            "ai_message": data.ai_message,
            "recommendations": [
                {"activity_id": r.activity_id}
                for r in data.recommendations
            ],
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(Config.BACKEND_CALLBACK_URL, json=payload)
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
