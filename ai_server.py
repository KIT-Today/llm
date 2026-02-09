# -*- coding: utf-8 -*-
"""
번아웃 감지 AI 서버 v2.0
=======================

POST /analyze : 분석 요청 -> 즉시 200 OK -> 백그라운드 분석 -> 콜백

실행: uvicorn ai_server:app --reload --port 8001
"""

import os
import httpx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from prompts import (
    PersonaType, 
    PERSONAS,
    PromptBuilder, 
    get_template_feedback,
)


# ============================================
# 설정
# ============================================

class Config:
    BACKEND_CALLBACK_URL = os.getenv("BACKEND_CALLBACK_URL", "http://127.0.0.1:8000/diaries/analysis-callback")
    MODEL_DIR = os.getenv("MODEL_DIR", ".")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIN_DIARY_COUNT_FOR_RECOMMENDATION = 3


# ============================================
# 활동 카테고리 매핑
# ============================================

BURNOUT_TO_ACTIVITY_CATEGORY = {
    "정서적_고갈": ["REST", "SMALL_WIN"],
    "좌절_압박": ["VENTILATION", "REST"],
    "부정적_대인관계": ["VENTILATION", "SMALL_WIN"],
    "자기비하": ["SMALL_WIN", "REST"],
}

ACTIVITY_CATEGORY_IDS = {
    "REST": [],
    "VENTILATION": [],
    "SMALL_WIN": [],
}


# ============================================
# 카테고리 정의
# ============================================

STAGE1_CATEGORIES = {0: "긍정", 1: "부정"}
STAGE2_CATEGORIES = {0: "정서적_고갈", 1: "좌절_압박", 2: "부정적_대인관계", 3: "자기비하"}

MBI_CATEGORY_MAP = {
    "긍정": "NONE",
    "정서적_고갈": "EMOTIONAL_EXHAUSTION",
    "좌절_압박": "FRUSTRATION_PRESSURE",
    "부정적_대인관계": "NEGATIVE_RELATIONSHIP",
    "자기비하": "SELF_DEPRECATION"
}

BURNOUT_KEYWORDS = {
    "긍정": {"keywords": ["좋다", "좋아", "행복", "기쁘", "뿌듯", "만족", "감사", "고맙", "다행", "홀가분"]},
    "정서적_고갈": {"keywords": ["지치", "피곤", "힘들", "무기력", "탈진", "녹초", "방전", "우울", "슬프", "귀찮"]},
    "좌절_압박": {"keywords": ["화나", "짜증", "열받", "분노", "억울", "압박", "스트레스", "답답", "한계"]},
    "부정적_대인관계": {"keywords": ["무시", "소외", "배신", "갈등", "서운", "외로", "실망", "오해"]},
    "자기비하": {"keywords": ["못하", "부족", "무능", "한심", "불안", "자책", "후회", "실패"]},
}


# ============================================
# Pydantic 모델
# ============================================

class DiaryHistory(BaseModel):
    diary_id: int
    content: Optional[str] = None
    keywords: Optional[Dict[str, Any]] = None
    created_at: str


class AnalyzeRequest(BaseModel):
    """백엔드 -> AI 서버 요청"""
    diary_id: int
    user_id: int
    persona: str = "warm_counselor"
    history: List[DiaryHistory]


class RecommendationItem(BaseModel):
    activity_id: int
    ai_message: str


class DiaryAnalysisResult(BaseModel):
    diary_id: int
    primary_emotion: str
    primary_score: float
    mbi_category: str
    keywords: List[str]


class AnalysisCallback(BaseModel):
    """AI 서버 -> 백엔드 콜백"""
    diary_id: int
    primary_emotion: str
    primary_score: float
    mbi_category: str
    emotion_probs: Dict[str, float]
    ai_message: str
    diary_analyses: List[DiaryAnalysisResult]
    recommendations: List[RecommendationItem]


# ============================================
# 분류 모델
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
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        if self._initialized:
            return
        
        print(f"모델 로딩 중... (Device: {Config.DEVICE})")
        
        self.kure = SentenceTransformer("nlpai-lab/KURE-v1", device=Config.DEVICE)
        
        s1_path = f"{Config.MODEL_DIR}/stage1_model.pt"
        s1_ckpt = torch.load(s1_path, map_location=Config.DEVICE, weights_only=False)
        self.stage1 = BurnoutClassifier(
            input_dim=s1_ckpt.get('embedding_dim', 1024),
            hidden_dim=s1_ckpt.get('hidden_dim', 256),
            num_classes=2
        ).to(Config.DEVICE)
        self.stage1.load_state_dict(s1_ckpt['model_state_dict'])
        self.stage1.eval()
        
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
        print("모델 로딩 완료!")
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        return self.kure.encode(text, convert_to_tensor=True).unsqueeze(0).to(Config.DEVICE)
    
    def predict_stage1(self, text: str) -> tuple:
        with torch.no_grad():
            emb = self._get_embedding(text)
            logits = self.stage1(emb)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return pred, probs
    
    def predict_stage2(self, text: str) -> tuple:
        with torch.no_grad():
            emb = self._get_embedding(text)
            logits = self.stage2(emb)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return pred, probs
    
    def extract_keywords(self, text: str, category: str, top_k: int = 3) -> List[str]:
        if category not in BURNOUT_KEYWORDS:
            return []
        keywords = BURNOUT_KEYWORDS[category]["keywords"]
        matched = [kw for kw in keywords if kw in text]
        return matched[:top_k]
    
    def analyze(self, text: str, keywords: Optional[Dict] = None) -> Dict:
        analysis_text = text or ""
        if keywords:
            keyword_text = " ".join([f"{k}: {v}" if isinstance(v, str) else str(v) for k, v in keywords.items()])
            analysis_text = f"{analysis_text} {keyword_text}".strip()
        
        if not analysis_text:
            return {
                "primary_emotion": "긍정", "primary_score": 0.5, "mbi_category": "NONE",
                "emotion_probs": {"긍정": 0.5, "부정": 0.5}, "burnout_category": None, "keywords": []
            }
        
        s1_pred, s1_probs = self.predict_stage1(analysis_text)
        primary_emotion = STAGE1_CATEGORIES[s1_pred]
        
        result = {
            "primary_emotion": primary_emotion,
            "primary_score": float(s1_probs[s1_pred]),
            "emotion_probs": {"긍정": float(s1_probs[0]), "부정": float(s1_probs[1])},
            "burnout_category": None, "mbi_category": "NONE", "keywords": []
        }
        
        if s1_pred == 1:
            s2_pred, s2_probs = self.predict_stage2(analysis_text)
            burnout_category = STAGE2_CATEGORIES[s2_pred]
            result["burnout_category"] = burnout_category
            result["mbi_category"] = MBI_CATEGORY_MAP[burnout_category]
            result["keywords"] = self.extract_keywords(analysis_text, burnout_category)
        
        return result


# ============================================
# 피드백 생성기 (템플릿 + LLM)
# ============================================

class FeedbackGenerator:
    def __init__(self, use_llm: bool = False, persona_type: PersonaType = PersonaType.WARM_COUNSELOR):
        self.use_llm = use_llm
        self.persona_type = persona_type
        self.prompt_builder = PromptBuilder(persona_type)
        self.generator = None
        self.tokenizer = None
        
        if use_llm:
            self._load_llm()
    
    def set_persona(self, persona_type: PersonaType):
        self.persona_type = persona_type
        self.prompt_builder.set_persona(persona_type)
    
    def _load_llm(self):
        """KoAlpaca LLM 로드"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
            print(f"LLM 로딩 중: {MODEL_NAME}")
            
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
            print("LLM 로딩 완료!")
        except Exception as e:
            print(f"LLM 로딩 실패, 템플릿 모드 사용: {e}")
            self.use_llm = False
    
    def generate(self, category: str, user_text: str = "", keywords: List[str] = None) -> str:
        if self.use_llm and self.generator:
            return self._generate_llm(category, user_text, keywords)
        return self._generate_template(category, keywords)
    
    def _generate_template(self, category: str, keywords: List[str] = None) -> str:
        return get_template_feedback(persona_type=self.persona_type, category=category, keywords=keywords)
    
    def _generate_llm(self, category: str, user_text: str, keywords: List[str]) -> str:
        """LLM으로 피드백 생성"""
        persona = PERSONAS[self.persona_type]
        
        prompt = f"""### 명령어:
당신은 '{persona.name}'입니다. {persona.description}
번아웃을 겪는 직장인에게 {persona.tone} 톤으로 2-3문장의 공감 메시지를 작성하세요.

규칙:
- {persona.tone} 톤 유지
- 감정을 인정하고 공감
- 강요하지 않고 부드럽게 제안
- 이모지 사용 금지

### 입력:
감정 상태: {category}
사용자 일기: "{user_text[:150] if user_text else '(내용 없음)'}"
주요 키워드: {', '.join(keywords) if keywords else '없음'}

### 응답:
"""
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated = result[0]['generated_text']
            response = generated.split("### 응답:")[-1].strip()
            
            # 줄바꿈 이후 잘라내기
            if "\n\n" in response:
                response = response.split("\n\n")[0].strip()
            
            # 응답이 너무 짧으면 템플릿으로 폴백
            if len(response) < 10:
                return self._generate_template(category, keywords)
            
            return response
            
        except Exception as e:
            print(f"LLM 생성 실패: {e}")
            return self._generate_template(category, keywords)


# ============================================
# 글로벌 인스턴스
# ============================================

analyzer: Optional[BurnoutAnalyzer] = None
feedback_gen: Optional[FeedbackGenerator] = None

PERSONA_MAP = {
    "warm_counselor": PersonaType.WARM_COUNSELOR,
    "practical_advisor": PersonaType.PRACTICAL_ADVISOR,
    "friendly_buddy": PersonaType.FRIENDLY_BUDDY,
    "calm_mentor": PersonaType.CALM_MENTOR,
    "cheerful_supporter": PersonaType.CHEERFUL_SUPPORTER,
}


# ============================================
# FastAPI 앱
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer, feedback_gen
    analyzer = BurnoutAnalyzer()
    analyzer.initialize()
    
    # LLM 사용 여부 환경변수로 설정 (USE_LLM=true)
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    print(f"\ud53c드백 모드: {'LLM (KoAlpaca)' if use_llm else '템플릿'}")
    feedback_gen = FeedbackGenerator(use_llm=use_llm)
    
    yield
    print("서버 종료")


app = FastAPI(
    title="번아웃 감지 AI 서버",
    description="한국형 번아웃 감정 분석 및 피드백 생성 API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ============================================
# API 엔드포인트
# ============================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Burnout Detection AI Server",
        "version": "2.0.0",
        "device": Config.DEVICE,
        "model_loaded": analyzer is not None and analyzer._initialized
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_diary(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    일기 분석 요청
    
    - diary_id: 오늘 일기 ID
    - user_id: 사용자 ID
    - persona: 피드백 말투 (warm_counselor, practical_advisor, friendly_buddy, calm_mentor, cheerful_supporter)
    - history: 2주치 일기 (오늘이 index 0)
    
    즉시 200 OK 반환 후 백그라운드에서 분석, 완료 시 콜백 전송
    """
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


async def process_analysis(diary_id: int, user_id: int, persona: str, history: List[DiaryHistory]):
    """백그라운드 분석"""
    try:
        print(f"분석 시작: diary_id={diary_id}, user_id={user_id}, persona={persona}")
        
        # 1. 페르소나 설정
        persona_type = PERSONA_MAP.get(persona, PersonaType.WARM_COUNSELOR)
        feedback_gen.set_persona(persona_type)
        
        # 2. 모든 일기 분석
        diary_analyses = []
        for diary in history:
            result = analyzer.analyze(diary.content or "", diary.keywords or {})
            diary_analyses.append(DiaryAnalysisResult(
                diary_id=diary.diary_id,
                primary_emotion=result["primary_emotion"],
                primary_score=round(result["primary_score"], 4),
                mbi_category=result["mbi_category"],
                keywords=result.get("keywords", [])
            ))
        
        # 3. 오늘 일기 분석
        today_diary = history[0]
        today_result = analyzer.analyze(today_diary.content or "", today_diary.keywords or {})
        
        category = "긍정" if today_result["primary_emotion"] == "긍정" else today_result.get("burnout_category", "정서적_고갈")
        
        # 4. 피드백 생성
        ai_message = feedback_gen.generate(
            category=category,
            user_text=today_diary.content or "",
            keywords=today_result.get("keywords", [])
        )
        
        # 5. 활동 추천 (3개 이상일 때만)
        recommendations = []
        if len(history) >= Config.MIN_DIARY_COUNT_FOR_RECOMMENDATION:
            recommendations = generate_recommendations(category, today_diary.content or "", today_result.get("keywords", []))
        
        # 6. 콜백 전송
        callback_data = AnalysisCallback(
            diary_id=diary_id,
            primary_emotion=today_result["primary_emotion"],
            primary_score=round(today_result["primary_score"], 4),
            mbi_category=today_result["mbi_category"],
            emotion_probs=today_result["emotion_probs"],
            ai_message=ai_message,
            diary_analyses=diary_analyses,
            recommendations=recommendations
        )
        
        await send_callback(callback_data)
        print(f"분석 완료: diary_id={diary_id}, 일기수={len(diary_analyses)}, 추천수={len(recommendations)}")
        
    except Exception as e:
        print(f"분석 실패: diary_id={diary_id}, error={e}")
        import traceback
        traceback.print_exc()


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
            ai_message = feedback_gen.generate(category=category, user_text=user_text, keywords=keywords)
            recommendations.append(RecommendationItem(activity_id=selected_id, ai_message=ai_message))
    
    return recommendations


async def send_callback(data: AnalysisCallback):
    """백엔드 콜백"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(Config.BACKEND_CALLBACK_URL, json=data.model_dump())
            if response.status_code == 200:
                print(f"콜백 성공: diary_id={data.diary_id}")
            else:
                print(f"콜백 실패: status={response.status_code}")
    except Exception as e:
        print(f"콜백 에러: {e}")


# ============================================
# 테스트/설정 엔드포인트
# ============================================

@app.post("/analyze/sync")
async def analyze_sync(request: AnalyzeRequest):
    """동기 분석 (테스트용) - 콜백 없이 바로 결과 반환"""
    if not request.history:
        raise HTTPException(status_code=400, detail="history가 비어있습니다.")
    
    persona_type = PERSONA_MAP.get(request.persona, PersonaType.WARM_COUNSELOR)
    feedback_gen.set_persona(persona_type)
    
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
    
    today_diary = request.history[0]
    today_result = analyzer.analyze(today_diary.content or "", today_diary.keywords or {})
    category = "긍정" if today_result["primary_emotion"] == "긍정" else today_result.get("burnout_category", "정서적_고갈")
    ai_message = feedback_gen.generate(category=category, user_text=today_diary.content or "", keywords=today_result.get("keywords", []))
    
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
        recommendations=recommendations
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
    """
    활동 ID 설정 (백엔드에서 호출)
    
    예시:
    {
        "REST": [1, 2, 3],
        "VENTILATION": [4, 5, 6],
        "SMALL_WIN": [7, 8, 9]
    }
    """
    global ACTIVITY_CATEGORY_IDS
    for category, ids in activities.items():
        if category in ACTIVITY_CATEGORY_IDS:
            ACTIVITY_CATEGORY_IDS[category] = ids
    return {"status": "updated", "activities": ACTIVITY_CATEGORY_IDS}


@app.get("/config/activities")
async def get_activity_ids():
    """현재 활동 ID 조회"""
    return {"activities": ACTIVITY_CATEGORY_IDS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8001, reload=True)
