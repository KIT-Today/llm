# -*- coding: utf-8 -*-
"""
감정 일치도 검사기
"""

from typing import Dict, Any

from config import Config
from constants import ENERGY_TO_EMOTION_MAP, DETAIL_KEYWORD_TO_EMOTION
from models import EmotionMatchResult
from prompts import PersonaType, PERSONAS


class EmotionMatchChecker:
    """사용자 선택 감정과 AI 분석 결과 비교"""
    
    def __init__(self, persona_type: PersonaType = PersonaType.WARM_COUNSELOR):
        self.persona_type = persona_type
    
    def set_persona(self, persona_type: PersonaType):
        """페르소나 변경"""
        self.persona_type = persona_type
    
    def check_match(self, user_keywords: Dict[str, Any], ai_result: Dict) -> EmotionMatchResult:
        """감정 일치도 검사"""
        # 사용자 선택에서 감정 추출
        user_emotion = self._extract_user_emotion(user_keywords)
        
        # AI 분석 결과에서 감정 추출
        ai_emotion = ai_result.get("burnout_category") or ai_result.get("primary_emotion", "긍정")
        
        # 일치도 계산
        match_score = self._calculate_match_score(user_emotion, ai_emotion)
        is_matched = match_score >= Config.EMOTION_MISMATCH_THRESHOLD
        
        # 불일치 시 힌트 메시지 생성
        hidden_emotion_hint = None
        if not is_matched:
            hidden_emotion_hint = self._generate_mismatch_hint(user_emotion, ai_emotion)
        
        return EmotionMatchResult(
            is_matched=is_matched,
            user_emotion=user_emotion,
            ai_emotion=ai_emotion,
            match_score=round(match_score, 2),
            hidden_emotion_hint=hidden_emotion_hint
        )
    
    def _extract_user_emotion(self, user_keywords: Dict[str, Any]) -> str:
        """사용자 선택에서 주요 감정 추출"""
        # 상세 키워드에서 감정 추출
        detail_keywords = user_keywords.get("detail_keywords", [])
        if isinstance(detail_keywords, list) and detail_keywords:
            for kw in detail_keywords:
                if kw in DETAIL_KEYWORD_TO_EMOTION:
                    return DETAIL_KEYWORD_TO_EMOTION[kw]
        
        # 에너지 분류에서 감정 추출
        energy_category = user_keywords.get("energy_category", "")
        if energy_category in ENERGY_TO_EMOTION_MAP:
            return ENERGY_TO_EMOTION_MAP[energy_category][0]
        
        return "긍정"
    
    def _calculate_match_score(self, user_emotion: str, ai_emotion: str) -> float:
        """일치도 점수 계산"""
        # 완전 일치
        if user_emotion == ai_emotion:
            return 1.0
        
        # 긍정 vs 부정 대분류 일치 확인
        user_is_positive = user_emotion == "긍정"
        ai_is_positive = ai_emotion == "긍정"
        
        if user_is_positive != ai_is_positive:
            return 0.2
        
        # 같은 부정 카테고리 내에서의 유사도
        burnout_similarity = {
            ("정서적_고갈", "좌절_압박"): 0.6,
            ("정서적_고갈", "자기비하"): 0.5,
            ("좌절_압박", "부정적_대인관계"): 0.6,
            ("부정적_대인관계", "자기비하"): 0.5,
            ("정서적_고갈", "부정적_대인관계"): 0.4,
            ("좌절_압박", "자기비하"): 0.5,
        }
        
        pair = tuple(sorted([user_emotion, ai_emotion]))
        return burnout_similarity.get(pair, 0.4)
    
    def _generate_mismatch_hint(self, user_emotion: str, ai_emotion: str) -> str:
        """불일치 시 힌트 메시지 생성"""
        emotion_korean = {
            "긍정": "긍정적인 감정",
            "정서적_고갈": "정서적 피로감",
            "좌절_압박": "좌절감이나 압박감",
            "부정적_대인관계": "대인관계에서의 어려움",
            "자기비하": "자신에 대한 부정적인 생각"
        }
        
        user_emotion_kr = emotion_korean.get(user_emotion, user_emotion)
        ai_emotion_kr = emotion_korean.get(ai_emotion, ai_emotion)
        
        if self.persona_type == PersonaType.FRIENDLY_BUDDY:
            return f"{user_emotion_kr}이라고 했는데, 혹시 {ai_emotion_kr}도 느끼고 있는 거 아니야? 괜찮아, 솔직해도 돼."
        elif self.persona_type == PersonaType.PRACTICAL_ADVISOR:
            return f"선택하신 '{user_emotion_kr}' 외에 '{ai_emotion_kr}'의 흔적도 보입니다. 다양한 감정이 섞여 있을 수 있어요."
        elif self.persona_type == PersonaType.CHEERFUL_SUPPORTER:
            return f"'{user_emotion_kr}'라고 하셨지만, '{ai_emotion_kr}'도 살짝 느껴지는 것 같아요! 감정은 복잡한 거니까요~"
        elif self.persona_type == PersonaType.CALM_MENTOR:
            return f"'{user_emotion_kr}' 뒤에 '{ai_emotion_kr}'이 숨어 있을 수 있습니다. 천천히 자신의 마음을 들여다보세요."
        else:  # WARM_COUNSELOR
            return f"'{user_emotion_kr}'라고 하셨지만, 기록에서는 '{ai_emotion_kr}'도 느껴져요. 혹시 숨겨진 감정이 있으신 건 아닐까요?"
