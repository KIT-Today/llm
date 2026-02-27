# -*- coding: utf-8 -*-
"""
Pydantic 모델 정의
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from error_codes import ErrorDetail


# ============================================
# 요청/응답 모델
# ============================================

class DiaryHistory(BaseModel):
    """일기 히스토리 항목"""
    diary_id: int
    content: Optional[str] = None
    keywords: Optional[Dict[str, Any]] = None
    created_at: str


class AnalyzeRequest(BaseModel):
    """백엔드 -> AI 서버 요청"""
    diary_id: int
    user_id: int
    persona: Any = "warm_counselor"  # str 또는 int
    history: List[DiaryHistory]


class RecommendationItem(BaseModel):
    """활동 추천 항목"""
    act_content: str            # 활동 내용 (예: "따뜻한 차/코코아 한 잔 마시기")
    act_category: str           # 활동 카테고리: REST / VENTILATION / SMALL_WIN
    is_active: bool             # 신체 활동 여부
    is_outdoor: bool            # 야외 활동 여부
    is_social: bool             # 사회적 교류 여부
    ai_message: str = ""        # AI 추천 멘트 (LLM 사용 시 개인화, 템플릿 모드 시 빈 문자열)


class DiaryAnalysisResult(BaseModel):
    """개별 일기 분석 결과"""
    diary_id: int
    primary_emotion: str
    primary_score: float
    mbi_category: str
    keywords: List[str]


class EmotionMatchResult(BaseModel):
    """감정 일치도 검사 결과"""
    is_matched: bool
    user_emotion: str
    ai_emotion: str
    match_score: float
    hidden_emotion_hint: Optional[str] = None


class MBIRiskItem(BaseModel):
    """MBI 단일 요소 위험도"""
    ratio: float                  # 현재 비율 (0.0 ~ 1.0)
    threshold: float              # 논문 기준 컷오프
    is_risk: bool                 # 위험군 여부
    contributing: List[str]       # 기여한 4카테고리 목록


class MBIAssessment(BaseModel):
    """
    MBI 3요소 위험도 평가
    근거: Lee et al. (2017) MBI 절단점 + Maslach & Jackson (1981) 정의
    4카테고리 → MBI 3요소 해석 레이어 (한국형 세분화 역매핑)
    """
    EE: MBIRiskItem   # 정서적 고갈 (Emotional Exhaustion)
    DP: MBIRiskItem   # 비인격화 (Depersonalization)
    PA: MBIRiskItem   # 자아 성취감 저하 (Reduced Personal Accomplishment)
    overall_risk: bool


class StatisticsInsight(BaseModel):
    """통계 인사이트"""
    period: str
    total_entries: int
    emotion_frequency: Dict[str, int]
    situation_frequency: Dict[str, int]
    top_keywords: List[str]
    burnout_trend: Dict[str, int]
    insight_messages: List[str]
    mbi_assessment: Optional[MBIAssessment] = None


class AnalysisCallback(BaseModel):
    """AI 서버 -> 백엔드 콜백"""
    diary_id: int
    primary_emotion: str
    primary_score: float
    mbi_category: str
    emotion_probs: Dict[str, Any]
    ai_message: str
    diary_analyses: List[DiaryAnalysisResult]
    recommendations: List[RecommendationItem]
    emotion_match: Optional[EmotionMatchResult] = None
    statistics_insight: Optional[StatisticsInsight] = None
    
    # 에러 정보
    success: bool = True
    errors: List[ErrorDetail] = []
    fallback_used: bool = False


# ============================================
# 피드백 요청/응답 모델
# ============================================

class FeedbackRecord(BaseModel):
    """단일 피드백 레코드 (백엔드 DB 1행)"""
    predicted_mbi_category: str          # AI가 예측한 카테고리
    is_correct: bool                     # 사용자 정오 확인
    satisfaction_score: int              # 만족도 1~5
    user_mbi_category: Optional[str] = None  # 틀렸을 때 사용자가 선택한 카테고리


class FeedbackBatchRequest(BaseModel):
    """백엔드 -> AI 서버 2주 배치 전송"""
    period_start: str                    # "2026-02-01"
    period_end: str                      # "2026-02-14"
    records: List[FeedbackRecord]


class FeedbackBatchResponse(BaseModel):
    """배치 수신 응답"""
    status: str
    received: int                        # 수신된 레코드 수
    total_accumulated: int               # 누적 전체 수
    model_accuracy: float                # 전체 정확도
    category_corrections: Dict[str, int] # 카테고리별 오답 수
