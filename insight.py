# -*- coding: utf-8 -*-
"""
통계 인사이트 생성기
"""

from typing import List, Optional
from collections import Counter

from config import Config
from models import DiaryHistory, DiaryAnalysisResult, StatisticsInsight
from prompts import PersonaType, PERSONAS


class StatisticsInsightGenerator:
    """기간별 통계 인사이트 생성"""
    
    def __init__(self, persona_type: PersonaType = PersonaType.WARM_COUNSELOR):
        self.persona_type = persona_type
    
    def set_persona(self, persona_type: PersonaType):
        """페르소나 변경"""
        self.persona_type = persona_type
    
    def generate(self, diary_analyses: List[DiaryAnalysisResult], history: List[DiaryHistory]) -> Optional[StatisticsInsight]:
        """통계 인사이트 생성"""
        if len(diary_analyses) < Config.MIN_DIARY_COUNT_FOR_INSIGHT:
            return None
        
        # 기간 판단
        period = "weekly" if len(diary_analyses) <= 7 else "monthly"
        
        # 감정 빈도 집계
        emotion_frequency = Counter()
        burnout_trend = Counter()
        
        for analysis in diary_analyses:
            if analysis.primary_emotion == "부정":
                burnout_trend[analysis.mbi_category] += 1
            emotion_frequency[analysis.primary_emotion] += 1
        
        # 키워드 빈도 집계
        all_keywords = []
        situation_frequency = Counter()
        
        for diary in history:
            if diary.keywords:
                detail_kws = diary.keywords.get("detail_keywords", [])
                if isinstance(detail_kws, list):
                    all_keywords.extend(detail_kws)
                
                situation = diary.keywords.get("situation", "")
                if situation:
                    situation_frequency[situation] += 1
        
        keyword_counter = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_counter.most_common(5)]
        
        # 인사이트 메시지 생성
        insight_messages = self._generate_insight_messages(
            emotion_frequency, situation_frequency, keyword_counter, burnout_trend, len(diary_analyses)
        )
        
        return StatisticsInsight(
            period=period,
            total_entries=len(diary_analyses),
            emotion_frequency=dict(emotion_frequency),
            situation_frequency=dict(situation_frequency),
            top_keywords=top_keywords,
            burnout_trend=dict(burnout_trend),
            insight_messages=insight_messages
        )
    
    def _generate_insight_messages(
        self,
        emotion_freq: Counter,
        situation_freq: Counter,
        keyword_freq: Counter,
        burnout_trend: Counter,
        total: int
    ) -> List[str]:
        """인사이트 메시지 생성"""
        messages = []
        
        # 1. 부정 감정 비율 체크
        negative_count = emotion_freq.get("부정", 0)
        if negative_count > 0:
            negative_ratio = negative_count / total
            if negative_ratio >= 0.7:
                messages.append(self._format_message(
                    f"이번 기간 기록의 {int(negative_ratio * 100)}%가 힘든 감정이었어요. 충분히 쉬어가도 괜찮아요."
                ))
        
        # 2. 반복되는 상황 키워드
        for situation, count in situation_freq.most_common(2):
            if count >= 3:
                messages.append(self._format_message(
                    f"'{situation}'을(를) {count}번이나 기록하셨네요. 환경적인 요인이 크게 작용하고 있는 것 같아요."
                ))
        
        # 3. 자주 등장하는 감정 키워드
        for keyword, count in keyword_freq.most_common(2):
            if count >= 3:
                messages.append(self._format_message(
                    f"'{keyword}'이(가) {count}번 등장했어요. 이 감정에 대해 조금 더 살펴보는 건 어떨까요?"
                ))
        
        # 4. 번아웃 카테고리 트렌드
        if burnout_trend:
            most_common_burnout = burnout_trend.most_common(1)[0]
            category, count = most_common_burnout
            category_korean = {
                "EMOTIONAL_EXHAUSTION": "정서적 피로",
                "FRUSTRATION_PRESSURE": "압박감",
                "NEGATIVE_RELATIONSHIP": "대인관계 어려움",
                "SELF_DEPRECATION": "자기 의심"
            }
            if count >= 2:
                messages.append(self._format_message(
                    f"'{category_korean.get(category, category)}'과 관련된 기록이 {count}번 있었어요. 이 부분에 특히 신경 써주세요."
                ))
        
        # 5. 긍정적인 변화 체크
        positive_count = emotion_freq.get("긍정", 0)
        if positive_count >= 2:
            messages.append(self._format_message(
                f"긍정적인 날도 {positive_count}번 있었어요! 작은 기쁨들을 놓치지 않고 계시네요."
            ))
        
        return messages[:4]
    
    def _format_message(self, base_message: str) -> str:
        """페르소나에 맞게 메시지 포맷팅"""
        if self.persona_type == PersonaType.FRIENDLY_BUDDY:
            return base_message.replace("어요.", "어!").replace("요.", "야.")
        elif self.persona_type == PersonaType.CHEERFUL_SUPPORTER:
            return base_message + " 💪"
        return base_message
