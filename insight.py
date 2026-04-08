# -*- coding: utf-8 -*-
"""
통계 인사이트 생성기
"""

from typing import List, Optional
from collections import Counter

from config import Config
from models import DiaryHistory, DiaryAnalysisResult, StatisticsInsight, MBIAssessment, MBIRiskItem
from prompts import PersonaType, PERSONAS


class StatisticsInsightGenerator:
    """기간별 통계 인사이트 생성"""

    def __init__(self, persona_type: PersonaType = PersonaType.WARM_COUNSELOR):
        self.persona_type = persona_type

    def set_persona(self, persona_type: PersonaType):
        self.persona_type = persona_type

    def generate(self, diary_analyses: List[DiaryAnalysisResult], history: List[DiaryHistory]) -> Optional[StatisticsInsight]:
        """통계 인사이트 생성"""
        if len(diary_analyses) < Config.MIN_DIARY_COUNT_FOR_INSIGHT:
            return None

        period = "weekly" if len(diary_analyses) <= 7 else "monthly"

        emotion_frequency: Counter = Counter()
        burnout_trend: Counter = Counter()

        for analysis in diary_analyses:
            emotion_frequency[analysis.primary_emotion] += 1
            if analysis.primary_emotion == "부정" and analysis.mbi_category != "NORMAL":
                burnout_trend[analysis.mbi_category] += 1

        # 키워드·상황 빈도 집계
        all_keywords = []
        situation_frequency: Counter = Counter()

        for diary in history:
            if diary.keywords:
                # 백엔드 keywords 형식: {"나의 유형": ..., "감정": ..., "상황/원인": ...}
                situation = diary.keywords.get("상황/원인", "") or diary.keywords.get("situation", "")
                if situation:
                    situation_frequency[situation] += 1

                # 구형 detail_keywords 호환
                detail_kws = diary.keywords.get("detail_keywords", [])
                if isinstance(detail_kws, list):
                    all_keywords.extend(detail_kws)

        keyword_counter = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_counter.most_common(5)]

        insight_messages = self._generate_insight_messages(
            emotion_frequency, situation_frequency, keyword_counter, burnout_trend, len(diary_analyses)
        )

        mbi_assessment = self._calculate_mbi_assessment(
            burnout_trend, emotion_frequency, len(diary_analyses)
        )

        return StatisticsInsight(
            period=period,
            total_entries=len(diary_analyses),
            emotion_frequency=dict(emotion_frequency),
            situation_frequency=dict(situation_frequency),
            top_keywords=top_keywords,
            burnout_trend=dict(burnout_trend),
            insight_messages=insight_messages,
            mbi_assessment=mbi_assessment
        )

    def _generate_insight_messages(
        self,
        emotion_freq: Counter,
        situation_freq: Counter,
        keyword_freq: Counter,
        burnout_trend: Counter,
        total: int
    ) -> List[str]:
        messages = []

        # 1. 부정 감정 비율
        negative_count = emotion_freq.get("부정", 0)
        if negative_count > 0:
            negative_ratio = negative_count / total
            if negative_ratio >= 0.7:
                messages.append(self._format_message(
                    f"이번 기간 기록의 {int(negative_ratio * 100)}%가 힘든 감정이었어요. 충분히 쉬어가도 괜찮아요."
                ))

        # 2. 반복되는 상황
        for situation, count in situation_freq.most_common(2):
            if count >= 3:
                messages.append(self._format_message(
                    f"'{situation}'을(를) {count}번이나 기록하셨네요. 환경적인 요인이 크게 작용하고 있는 것 같아요."
                ))

        # 3. 자주 등장하는 키워드
        for keyword, count in keyword_freq.most_common(2):
            if count >= 3:
                messages.append(self._format_message(
                    f"'{keyword}'이(가) {count}번 등장했어요. 이 감정에 대해 조금 더 살펴보는 건 어떨까요?"
                ))

        # 4. 번아웃 카테고리 트렌드 (한국어 키 기준)
        CATEGORY_KOREAN = {
            "정서적_고갈": "정서적 피로",
            "좌절_압박": "압박감",
            "부정적_대인관계": "대인관계 어려움",
            "자기비하": "자기 의심",
        }
        if burnout_trend:
            category, count = burnout_trend.most_common(1)[0]
            if count >= 2:
                label = CATEGORY_KOREAN.get(category, category)
                messages.append(self._format_message(
                    f"'{label}'과 관련된 기록이 {count}번 있었어요. 이 부분에 특히 신경 써주세요."
                ))

        # 5. 긍정적인 날
        positive_count = emotion_freq.get("긍정", 0)
        if positive_count >= 2:
            messages.append(self._format_message(
                f"긍정적인 날도 {positive_count}번 있었어요! 작은 기쁨들을 놓치지 않고 계시네요."
            ))

        return messages[:4]

    def _calculate_mbi_assessment(
        self,
        burnout_trend: Counter,
        emotion_frequency: Counter,
        total: int
    ) -> MBIAssessment:
        """
        MBI 3요소 위험도 계산 (한국어 카테고리 키 기준)

        매핑:
          EE ← 정서적_고갈
          DP ← 좌절_압박 + 부정적_대인관계
          PA ← 긍정 비율 부재 + 자기비하
        """
        # EE
        ee_count = burnout_trend.get("정서적_고갈", 0)
        ee_ratio = round(ee_count / total, 4) if total > 0 else 0.0
        EE_THRESHOLD = 0.55

        # DP
        dp_count = burnout_trend.get("좌절_압박", 0) + burnout_trend.get("부정적_대인관계", 0)
        dp_ratio = round(dp_count / total, 4) if total > 0 else 0.0
        DP_THRESHOLD = 0.36

        # PA
        positive_count = emotion_frequency.get("긍정", 0)
        pa_ratio = round(positive_count / total, 4) if total > 0 else 0.0
        PA_THRESHOLD = 0.68
        self_dep_ratio = burnout_trend.get("자기비하", 0) / total if total > 0 else 0.0
        pa_is_risk = (pa_ratio < PA_THRESHOLD) or (self_dep_ratio > 0.20)

        overall_risk = (
            ee_ratio >= EE_THRESHOLD or
            dp_ratio >= DP_THRESHOLD or
            pa_is_risk
        )

        return MBIAssessment(
            EE=MBIRiskItem(
                ratio=ee_ratio,
                threshold=EE_THRESHOLD,
                is_risk=ee_ratio >= EE_THRESHOLD,
                contributing=["정서적_고갈"]
            ),
            DP=MBIRiskItem(
                ratio=dp_ratio,
                threshold=DP_THRESHOLD,
                is_risk=dp_ratio >= DP_THRESHOLD,
                contributing=["좌절_압박", "부정적_대인관계"]
            ),
            PA=MBIRiskItem(
                ratio=pa_ratio,
                threshold=PA_THRESHOLD,
                is_risk=pa_is_risk,
                contributing=["긍정_부재", "자기비하"]
            ),
            overall_risk=overall_risk
        )

    def _format_message(self, base_message: str) -> str:
        if self.persona_type == PersonaType.FRIENDLY_BUDDY:
            return base_message.replace("어요.", "어!").replace("요.", "야.")
        elif self.persona_type == PersonaType.CHEERFUL_SUPPORTER:
            return base_message + " 💪"
        return base_message
