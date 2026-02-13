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
    activity_id: int
    ai_message: str


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


class StatisticsInsight(BaseModel):
    """통계 인사이트"""
    period: str
    total_entries: int
    emotion_frequency: Dict[str, int]
    situation_frequency: Dict[str, int]
    top_keywords: List[str]
    burnout_trend: Dict[str, int]
    insight_messages: List[str]


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
    emotion_match: Optional[EmotionMatchResult] = None
    statistics_insight: Optional[StatisticsInsight] = None
    
    # 에러 정보
    success: bool = True
    errors: List[ErrorDetail] = []
    fallback_used: bool = False
