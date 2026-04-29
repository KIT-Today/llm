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
    """일기 히스토리 항목

    백엔드 명세 2-5-0 기준:
    - type="CURRENT"  : 오늘 작성한 일기. content + keywords 포함.
    - type="PAST_ANALYSIS" : 과거 분석 완료 일기. 분석 결과만 포함 (content/keywords 없음).

    모든 필드를 Optional로 잡아 type에 따라 유연하게 처리.
    type 필드가 없는 구버전 요청도 하위 호환으로 수용 (content 있으면 CURRENT 취급).
    """
    diary_id: int
    created_at: str

    # CURRENT/PAST_ANALYSIS 구분
    type: Optional[str] = None

    # CURRENT 전용
    content: Optional[str] = None
    keywords: Optional[Dict[str, Any]] = None

    # PAST_ANALYSIS 전용 (이미 분석된 결과)
    primary_emotion: Optional[str] = None
    primary_score: Optional[float] = None
    mbi_category: Optional[str] = None  # 영문 (EMOTIONAL_EXHAUSTION 등) 또는 "NORMAL"/"NONE"
    emotion_probs: Optional[Dict[str, Any]] = None

    def is_current(self) -> bool:
        """오늘 작성된 일기인지 판정. type 필드 우선, 없으면 content 존재 여부로 폴백."""
        if self.type:
            return self.type == "CURRENT"
        return self.content is not None


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
    """단일 피드백 레코드"""
    predicted_mbi_category: str  # AI가 예측한 카테고리
    ai_message_rating: int       # AI 메시지 만족도 1~5
    mbi_category_rating: int     # MBI 카테고리 만족도 1~5


class FeedbackBatchRequest(BaseModel):
    """백엔드 -> AI 서버 배치 전송"""
    feedbacks: List[FeedbackRecord]


class FeedbackBatchResponse(BaseModel):
    """배치 수신 응답"""
    status: str
    received: int                        # 수신된 레코드 수
    total_accumulated: int               # 누적 전체 수
    avg_ai_message_rating: float         # 이번 배치 AI 메시지 평균 평점
    avg_mbi_category_rating: float       # 이번 배치 카테고리 평균 평점
    low_mbi_by_category: Dict[str, int]  # 카테고리별 낮은 평점(1~2) 수
