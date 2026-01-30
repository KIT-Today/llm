# -*- coding: utf-8 -*-
"""
🔥 번아웃 감지 AI 서버
=======================

백엔드 API 명세에 맞춘 FastAPI 서버
- POST /analyze : 분석 요청 받고 즉시 200 OK
- 백그라운드에서 분석 후 콜백으로 결과 전송

실행: uvicorn ai_server:app --reload --port 8001
"""

import os
import asyncio
import httpx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import random
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# 프롬프트 모듈 임포트
from prompts import (
    PersonaType, 
    PERSONAS,
    PromptBuilder, 
    get_template_feedback,
    get_persona_by_preference,
    CATEGORY_CONTEXT,
)

# ============================================
# 설정
# ============================================

class Config:
    # 백엔드 콜백 URL (배포 시 변경)
    BACKEND_CALLBACK_URL = os.getenv(
        "BACKEND_CALLBACK_URL", 
        "http://127.0.0.1:8000/diaries/analysis-callback"
    )
    
    # 모델 경로
    MODEL_DIR = os.getenv("MODEL_DIR", ".")
    
    # 디바이스
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 솔루션 활동 ID 매핑 (activities 테이블 기준)
    # TODO: 실제 DB의 activity_id에 맞게 수정 필요
    ACTIVITY_IDS = {
        "정서적_고갈": [1, 2, 3],      # 휴식, 명상 등
        "좌절_압박": [4, 5, 6],        # 스트레스 해소 등
        "부정적_대인관계": [7, 8, 9],   # 소통, 관계 회복 등
        "자기비하": [10, 11, 12],      # 자기 긍정, 성취감 등
    }


# ============================================
# 카테고리 및 키워드 정의
# ============================================

STAGE1_CATEGORIES = {0: "긍정", 1: "부정"}
STAGE2_CATEGORIES = {0: "정서적_고갈", 1: "좌절_압박", 2: "부정적_대인관계", 3: "자기비하"}

# MBI 카테고리 매핑 (백엔드 명세에 맞춤)
MBI_CATEGORY_MAP = {
    "긍정": "NONE",
    "정서적_고갈": "EMOTIONAL_EXHAUSTION",
    "좌절_압박": "FRUSTRATION_PRESSURE",
    "부정적_대인관계": "NEGATIVE_RELATIONSHIP",
    "자기비하": "SELF_DEPRECATION"
}

BURNOUT_KEYWORDS = {
    "긍정": {"keywords": ["좋다", "좋아", "행복", "기쁘", "뿌듯", "만족", "감사", "고맙", "다행", "홀가분", "상쾌", "힐링", "편안", "여유", "성공", "달성", "완료", "끝났", "칭찬", "인정", "보람", "즐겁", "신나", "설레", "기대", "희망", "웃"]},
    "부정": {"keywords": ["힘들", "지치", "피곤", "싫", "짜증", "화나", "억울", "슬프", "우울", "불안", "걱정", "무섭", "두렵", "외롭", "서운", "실망", "후회", "미안"]},
    "정서적_고갈": {"keywords": ["지치", "피곤", "힘들", "무기력", "탈진", "녹초", "방전", "지쳐", "의욕", "에너지", "기운", "무거", "공허", "텅", "비어", "메마르", "번아웃", "우울", "슬프", "눈물", "허무", "무의미", "싫어", "귀찮"]},
    "좌절_압박": {"keywords": ["화나", "화가", "짜증", "열받", "빡치", "분노", "억울", "불공평", "압박", "스트레스", "마감", "답답", "미치", "폭발", "한계", "못참", "왜", "도대체", "짓눌", "감당", "부담", "실적", "안되", "안풀"]},
    "부정적_대인관계": {"keywords": ["무시", "소외", "따돌", "왕따", "배신", "뒷담", "험담", "갈등", "싸우", "다투", "틀어", "소문", "오해", "믿었", "실망", "서운", "혼자", "외로", "편없", "거절", "빼고", "안끼", "정치", "눈치"]},
    "자기비하": {"keywords": ["못하", "못난", "부족", "무능", "한심", "자격", "불안", "걱정", "자책", "죄책", "잘못", "내탓", "미안", "후회", "열등", "비교", "왜나만", "자신없", "두렵", "무섭", "실패", "망", "가치없", "쓸모없"]},
}


# ============================================
# Pydantic 모델 (Request/Response)
# ============================================

class DiaryHistory(BaseModel):
    diary_id: int
    content: Optional[str] = None
    keywords: Optional[Dict[str, Any]] = None
    created_at: str

class AnalyzeRequest(BaseModel):
    """백엔드 → AI 서버 분석 요청"""
    diary_id: int
    user_id: int
    history: List[DiaryHistory]

class RecommendationItem(BaseModel):
    """솔루션 추천 아이템"""
    activity_id: int
    ai_message: str

class AnalysisCallback(BaseModel):
    """AI 서버 → 백엔드 콜백 응답"""
    diary_id: int
    primary_emotion: str          # "긍정" or "부정"
    primary_score: float          # 신뢰도 (0~1)
    mbi_category: str             # NONE, EMOTIONAL_EXHAUSTION 등
    emotion_probs: Dict[str, float]
    recommendations: List[RecommendationItem]


# ============================================
# 분류 모델 정의
# ============================================

class BurnoutClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# ============================================
# AI 분석 엔진
# ============================================

class BurnoutAnalyzer:
    """번아웃 분석 엔진 - 싱글톤으로 모델 유지"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        if self._initialized:
            return
        
        print(f"🚀 모델 로딩 중... (Device: {Config.DEVICE})")
        
        # KURE 임베딩 모델
        print("  📥 KURE 로딩...")
        self.kure = SentenceTransformer("nlpai-lab/KURE-v1", device=Config.DEVICE)
        
        # Stage 1 모델 (긍정/부정)
        print("  📥 Stage 1 모델 로딩...")
        s1_path = f"{Config.MODEL_DIR}/stage1_model.pt"
        s1_ckpt = torch.load(s1_path, map_location=Config.DEVICE, weights_only=False)
        self.stage1 = BurnoutClassifier(
            input_dim=s1_ckpt.get('embedding_dim', 1024),
            hidden_dim=s1_ckpt.get('hidden_dim', 256),
            num_classes=2
        ).to(Config.DEVICE)
        self.stage1.load_state_dict(s1_ckpt['model_state_dict'])
        self.stage1.eval()
        
        # Stage 2 모델 (4개 번아웃 카테고리)
        print("  📥 Stage 2 모델 로딩...")
        s2_path = f"{Config.MODEL_DIR}/stage2_model.pt"
        s2_ckpt = torch.load(s2_path, map_location=Config.DEVICE, weights_only=False)
        self.stage2 = BurnoutClassifier(
            input_dim=s2_ckpt.get('embedding_dim', 1024),
            hidden_dim=s2_ckpt.get('hidden_dim', 256),
            num_classes=4
        ).to(Config.DEVICE)
        self.stage2.load_state_dict(s2_ckpt['model_state_dict'])
        self.stage2.eval()
        
        self._initialized = True
        print("✅ 모델 로딩 완료!")
    
    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[가-힣]+", text)
        return [t for t in tokens if len(t) >= 2]
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        return self.kure.encode(text, convert_to_tensor=True).unsqueeze(0).to(Config.DEVICE)
    
    def predict_stage1(self, text: str) -> tuple:
        """1단계: 긍정/부정 분류"""
        with torch.no_grad():
            emb = self._get_embedding(text)
            logits = self.stage1(emb)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return pred, probs
    
    def predict_stage2(self, text: str) -> tuple:
        """2단계: 번아웃 카테고리 분류"""
        with torch.no_grad():
            emb = self._get_embedding(text)
            logits = self.stage2(emb)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return pred, probs
    
    def extract_keywords(self, text: str, category: str, top_k: int = 3) -> List[str]:
        """텍스트에서 해당 카테고리의 주요 키워드 추출"""
        if category not in BURNOUT_KEYWORDS:
            return []
        
        keywords = BURNOUT_KEYWORDS[category]["keywords"]
        matched = [kw for kw in keywords if kw in text]
        return matched[:top_k]
    
    def analyze(self, text: str, keywords: Optional[Dict] = None) -> Dict:
        """
        전체 분석 실행
        - text: 일기 내용
        - keywords: 사용자가 선택한 키워드 (optional)
        """
        # 텍스트가 비어있으면 키워드 기반으로 처리
        analysis_text = text or ""
        if keywords:
            # 키워드를 텍스트로 변환하여 추가
            keyword_text = " ".join([
                f"{k}: {v}" if isinstance(v, str) else str(v) 
                for k, v in keywords.items()
            ])
            analysis_text = f"{analysis_text} {keyword_text}".strip()
        
        if not analysis_text:
            # 분석할 내용이 없으면 기본값 반환
            return {
                "primary_emotion": "긍정",
                "primary_score": 0.5,
                "mbi_category": "NONE",
                "emotion_probs": {"긍정": 0.5, "부정": 0.5},
                "burnout_category": None,
                "burnout_probs": {},
                "keywords": []
            }
        
        # Stage 1: 긍정/부정 분류
        s1_pred, s1_probs = self.predict_stage1(analysis_text)
        primary_emotion = STAGE1_CATEGORIES[s1_pred]
        primary_score = float(s1_probs[s1_pred])
        
        result = {
            "primary_emotion": primary_emotion,
            "primary_score": primary_score,
            "emotion_probs": {
                "긍정": float(s1_probs[0]),
                "부정": float(s1_probs[1])
            },
            "burnout_category": None,
            "burnout_probs": {},
            "keywords": []
        }
        
        # 부정이면 Stage 2 실행
        if s1_pred == 1:  # 부정
            s2_pred, s2_probs = self.predict_stage2(analysis_text)
            burnout_category = STAGE2_CATEGORIES[s2_pred]
            
            result["burnout_category"] = burnout_category
            result["burnout_probs"] = {
                STAGE2_CATEGORIES[i]: float(p) for i, p in enumerate(s2_probs)
            }
            result["mbi_category"] = MBI_CATEGORY_MAP[burnout_category]
            result["keywords"] = self.extract_keywords(analysis_text, burnout_category)
        else:
            result["mbi_category"] = "NONE"
        
        return result


# ============================================
# 피드백 생성기 (LLM 또는 템플릿) - prompts.py 모듈 활용
# ============================================

class FeedbackGenerator:
    """
    AI 피드백 메시지 생성
    - 템플릿 기반 (빠름, 기본) - prompts.py의 FEEDBACK_TEMPLATES 사용
    - LLM 기반 (느림, 고품질) - prompts.py의 PromptBuilder 사용
    
    5가지 페르소나 지원:
    - WARM_COUNSELOR: 따뜻한 상담사
    - PRACTICAL_ADVISOR: 실용적 조언자  
    - FRIENDLY_BUDDY: 친근한 친구
    - CALM_MENTOR: 차분한 멘토
    - CHEERFUL_SUPPORTER: 밝은 응원단
    """
    
    def __init__(
        self, 
        use_llm: bool = False, 
        persona_type: PersonaType = PersonaType.WARM_COUNSELOR
    ):
        self.use_llm = use_llm
        self.persona_type = persona_type
        self.prompt_builder = PromptBuilder(persona_type)
        self.generator = None
        self.tokenizer = None
        
        if use_llm:
            self._load_llm()
    
    def set_persona(self, persona_type: PersonaType):
        """페르소나 변경"""
        self.persona_type = persona_type
        self.prompt_builder.set_persona(persona_type)
    
    def _load_llm(self):
        """LLM 모델 로드 (선택적)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
            print(f"📥 LLM 로딩 중: {MODEL_NAME}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            print("✅ LLM 로딩 완료!")
        except Exception as e:
            print(f"⚠️ LLM 로딩 실패, 템플릿 모드 사용: {e}")
            self.use_llm = False
    
    def generate(
        self, 
        category: str, 
        user_text: str = "", 
        keywords: List[str] = None,
        activity_name: str = "",
        user_preference: dict = None,
    ) -> str:
        """
        피드백 생성
        
        Args:
            category: 감정 카테고리 (정서적_고갈, 좌절_압박 등)
            user_text: 사용자 일기 내용
            keywords: 추출된 키워드 목록
            activity_name: 추천 활동 이름
            user_preference: 사용자 설문 결과 (페르소나 자동 선택용)
        """
        # 사용자 설문 결과가 있으면 페르소나 자동 선택
        if user_preference:
            auto_persona = get_persona_by_preference(user_preference)
            self.set_persona(auto_persona)
        
        if self.use_llm and self.generator:
            return self._generate_llm(category, user_text, keywords, activity_name)
        else:
            return self._generate_template(category, keywords)
    
    def _generate_template(self, category: str, keywords: List[str] = None) -> str:
        """템플릿 기반 피드백 - prompts.py 활용"""
        return get_template_feedback(
            persona_type=self.persona_type,
            category=category,
            keywords=keywords
        )
    
    def _generate_llm(
        self, 
        category: str, 
        user_text: str, 
        keywords: List[str],
        activity_name: str
    ) -> str:
        """LLM 기반 피드백 - prompts.py의 PromptBuilder 활용"""
        
        # 프롬프트 생성
        prompt = self.prompt_builder.build_feedback_prompt(
            category=category,
            user_text=user_text,
            keywords=keywords,
            activity_name=activity_name,
        )
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated = result[0]['generated_text']
            response = generated.split("### 응답")[-1].strip()
            
            # 줄바꿈 이후 잘라내기
            if "\n\n" in response:
                response = response.split("\n\n")[0].strip()
            
            # 응답이 너무 짧거나 이상하면 템플릿으로 폴백
            if len(response) < 10:
                return self._generate_template(category, keywords)
            
            return response
            
        except Exception as e:
            print(f"⚠️ LLM 생성 실패: {e}")
            return self._generate_template(category, keywords)


# ============================================
# 글로벌 인스턴스
# ============================================

analyzer: Optional[BurnoutAnalyzer] = None
feedback_gen: Optional[FeedbackGenerator] = None


# ============================================
# FastAPI 앱
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드"""
    global analyzer, feedback_gen
    
    analyzer = BurnoutAnalyzer()
    analyzer.initialize()
    
    # 피드백 생성기 (LLM 사용 여부는 환경변수로)
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    default_persona = os.getenv("DEFAULT_PERSONA", "warm_counselor")
    
    # 페르소나 문자열 매핑
    persona_map = {
        "warm_counselor": PersonaType.WARM_COUNSELOR,
        "practical_advisor": PersonaType.PRACTICAL_ADVISOR,
        "friendly_buddy": PersonaType.FRIENDLY_BUDDY,
        "calm_mentor": PersonaType.CALM_MENTOR,
        "cheerful_supporter": PersonaType.CHEERFUL_SUPPORTER,
    }
    persona_type = persona_map.get(default_persona, PersonaType.WARM_COUNSELOR)
    
    feedback_gen = FeedbackGenerator(use_llm=use_llm, persona_type=persona_type)
    
    yield
    
    # 종료 시 정리
    print("👋 서버 종료")


app = FastAPI(
    title="번아웃 감지 AI 서버",
    description="한국형 번아웃 감정 분석 및 피드백 생성 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API 엔드포인트
# ============================================

@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "service": "Burnout Detection AI Server",
        "device": Config.DEVICE,
        "model_loaded": analyzer is not None and analyzer._initialized
    }


@app.get("/health")
async def health_check():
    """헬스체크"""
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_diary(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    일기 분석 요청 (백엔드 → AI 서버)
    
    - 즉시 200 OK 반환
    - 백그라운드에서 분석 후 콜백으로 결과 전송
    """
    # 입력 검증
    if not request.history:
        raise HTTPException(status_code=400, detail="history가 비어있습니다.")
    
    # 백그라운드 태스크로 분석 실행
    background_tasks.add_task(
        process_analysis,
        diary_id=request.diary_id,
        user_id=request.user_id,
        history=request.history
    )
    
    return {"status": "accepted", "message": "분석이 시작되었습니다."}


async def process_analysis(diary_id: int, user_id: int, history: List[DiaryHistory]):
    """
    백그라운드 분석 처리
    """
    try:
        print(f"🔍 분석 시작: diary_id={diary_id}, user_id={user_id}")
        
        # 최신 일기 (첫 번째 항목)
        latest_diary = history[0]
        content = latest_diary.content or ""
        keywords = latest_diary.keywords or {}
        
        # 분석 실행
        analysis_result = analyzer.analyze(content, keywords)
        
        # 솔루션 추천 생성
        recommendations = generate_recommendations(
            category=analysis_result.get("burnout_category") or "긍정",
            user_text=content,
            keywords=analysis_result.get("keywords", [])
        )
        
        # 콜백 데이터 구성
        callback_data = AnalysisCallback(
            diary_id=diary_id,
            primary_emotion=analysis_result["primary_emotion"],
            primary_score=round(analysis_result["primary_score"], 4),
            mbi_category=analysis_result["mbi_category"],
            emotion_probs=analysis_result["emotion_probs"],
            recommendations=recommendations
        )
        
        # 백엔드로 콜백 전송
        await send_callback(callback_data)
        
        print(f"✅ 분석 완료: diary_id={diary_id}, category={analysis_result['mbi_category']}")
        
    except Exception as e:
        print(f"❌ 분석 실패: diary_id={diary_id}, error={e}")
        import traceback
        traceback.print_exc()


def generate_recommendations(
    category: str, 
    user_text: str, 
    keywords: List[str]
) -> List[RecommendationItem]:
    """
    솔루션 추천 생성
    """
    recommendations = []
    
    # 긍정이면 추천 없음
    if category == "긍정" or category is None:
        return recommendations
    
    # 해당 카테고리의 활동 ID 가져오기
    activity_ids = Config.ACTIVITY_IDS.get(category, [1, 2, 3])
    
    # 최대 3개 추천
    selected_ids = random.sample(activity_ids, min(3, len(activity_ids)))
    
    for activity_id in selected_ids:
        # AI 메시지 생성
        ai_message = feedback_gen.generate(
            category=category,
            user_text=user_text,
            keywords=keywords,
            activity_name=""  # TODO: 실제 활동명 조회
        )
        
        recommendations.append(RecommendationItem(
            activity_id=activity_id,
            ai_message=ai_message
        ))
    
    return recommendations


async def send_callback(data: AnalysisCallback):
    """
    백엔드로 분석 결과 콜백 전송
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                Config.BACKEND_CALLBACK_URL,
                json=data.model_dump()
            )
            
            if response.status_code == 200:
                print(f"📤 콜백 전송 성공: diary_id={data.diary_id}")
            else:
                print(f"⚠️ 콜백 전송 실패: status={response.status_code}, body={response.text}")
                
    except Exception as e:
        print(f"❌ 콜백 전송 에러: {e}")


# ============================================
# 테스트용 엔드포인트
# ============================================

@app.post("/analyze/sync")
async def analyze_sync(request: AnalyzeRequest):
    """
    동기 분석 (테스트용)
    - 분석 완료 후 결과 직접 반환
    """
    if not request.history:
        raise HTTPException(status_code=400, detail="history가 비어있습니다.")
    
    latest_diary = request.history[0]
    content = latest_diary.content or ""
    keywords = latest_diary.keywords or {}
    
    # 분석
    analysis_result = analyzer.analyze(content, keywords)
    
    # 추천 생성
    recommendations = generate_recommendations(
        category=analysis_result.get("burnout_category") or "긍정",
        user_text=content,
        keywords=analysis_result.get("keywords", [])
    )
    
    return AnalysisCallback(
        diary_id=request.diary_id,
        primary_emotion=analysis_result["primary_emotion"],
        primary_score=round(analysis_result["primary_score"], 4),
        mbi_category=analysis_result["mbi_category"],
        emotion_probs=analysis_result["emotion_probs"],
        recommendations=recommendations
    )


@app.post("/test/feedback")
async def test_feedback(
    category: str = "정서적_고갈",
    text: str = "오늘 너무 힘들었어",
    persona: str = "warm_counselor"
):
    """
    피드백 생성 테스트
    
    persona 옵션:
    - warm_counselor: 따뜻한 상담사
    - practical_advisor: 실용적 조언자
    - friendly_buddy: 친근한 친구
    - calm_mentor: 차분한 멘토
    - cheerful_supporter: 밝은 응원단
    """
    global feedback_gen
    
    # 페르소나 문자열 매핑
    persona_map = {
        "warm_counselor": PersonaType.WARM_COUNSELOR,
        "practical_advisor": PersonaType.PRACTICAL_ADVISOR,
        "friendly_buddy": PersonaType.FRIENDLY_BUDDY,
        "calm_mentor": PersonaType.CALM_MENTOR,
        "cheerful_supporter": PersonaType.CHEERFUL_SUPPORTER,
    }
    
    persona_type = persona_map.get(persona, PersonaType.WARM_COUNSELOR)
    feedback_gen.set_persona(persona_type)
    
    feedback = feedback_gen.generate(
        category=category,
        user_text=text,
        keywords=["지침", "힘듦"]
    )
    
    persona_info = PERSONAS[persona_type]
    
    return {
        "category": category, 
        "persona": {
            "type": persona,
            "name": persona_info.name,
            "tone": persona_info.tone,
        },
        "feedback": feedback
    }


@app.get("/personas")
async def list_all_personas():
    """사용 가능한 모든 페르소나 목록"""
    from prompts import list_personas
    return {"personas": list_personas()}


# ============================================
# 실행
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ai_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
