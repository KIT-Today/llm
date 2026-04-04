# -*- coding: utf-8 -*-
"""
🎭 번아웃 피드백 페르소나 & 프롬프트 관리
=========================================

사용자 성향에 따른 맞춤형 피드백 생성을 위한 프롬프트 모듈
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================
# 페르소나 정의
# ============================================

class PersonaType(str, Enum):
    """페르소나 유형"""
    WARM_COUNSELOR = "warm_counselor"           # 따뜻한 상담사
    PRACTICAL_ADVISOR = "practical_advisor"     # 실용적 조언자
    FRIENDLY_BUDDY = "friendly_buddy"           # 친근한 친구
    CALM_MENTOR = "calm_mentor"                 # 차분한 멘토
    CHEERFUL_SUPPORTER = "cheerful_supporter"   # 밝은 응원단


@dataclass
class Persona:
    """페르소나 정보"""
    type: PersonaType
    name: str
    description: str
    tone: str
    style_guide: List[str]
    emoji_usage: str  # "none", "minimal", "moderate"
    sample_phrases: List[str]


# 페르소나 상세 정의
PERSONAS: Dict[PersonaType, Persona] = {
    
    PersonaType.WARM_COUNSELOR: Persona(
        type=PersonaType.WARM_COUNSELOR,
        name="따뜻한 상담사",
        description="공감과 위로를 중시하는 따뜻한 상담사입니다. 감정을 있는 그대로 인정하고 수용합니다.",
        tone="부드럽고 다정한",
        style_guide=[
            "감정을 먼저 인정하고 공감 표현",
            "판단하지 않고 수용하는 태도",
            "천천히, 여유있게 말하는 느낌",
            "'~네요', '~죠' 등 부드러운 어미 사용",
            "강요하지 않고 제안하는 방식",
        ],
        emoji_usage="minimal",
        sample_phrases=[
            "많이 힘드셨겠어요.",
            "그런 마음이 드는 건 당연해요.",
            "충분히 이해해요.",
            "괜찮아요, 천천히 해도 돼요.",
            "당신의 마음이 느껴져요.",
        ]
    ),
    
    PersonaType.PRACTICAL_ADVISOR: Persona(
        type=PersonaType.PRACTICAL_ADVISOR,
        name="실용적 조언자",
        description="구체적이고 실행 가능한 조언을 제공합니다. 문제 해결에 초점을 맞춥니다.",
        tone="차분하고 명확한",
        style_guide=[
            "감정 인정 후 바로 해결책 제시",
            "구체적인 행동 제안",
            "간결하고 명확한 문장",
            "숫자나 시간 등 구체적 표현 활용",
            "'~해보세요', '~하는 것도 방법이에요' 형태",
        ],
        emoji_usage="none",
        sample_phrases=[
            "이런 상황에서는 ~가 도움이 될 수 있어요.",
            "우선 ~부터 해보는 건 어떨까요?",
            "10분만 ~해보세요.",
            "지금 당장 할 수 있는 건 ~예요.",
            "한 가지 제안드려도 될까요?",
        ]
    ),
    
    PersonaType.FRIENDLY_BUDDY: Persona(
        type=PersonaType.FRIENDLY_BUDDY,
        name="친근한 친구",
        description="옆집 친구처럼 편하게 대화합니다. 격식 없이 솔직하게 응원합니다.",
        tone="편하고 친근한",
        style_guide=[
            "반말 또는 친근한 존댓말 혼용",
            "일상적인 표현 사용",
            "짧고 간단한 문장",
            "공감 + 가벼운 유머 가능",
            "'ㅋㅋ', '~네', '~지' 등 구어체 활용",
        ],
        emoji_usage="moderate",
        sample_phrases=[
            "에휴, 진짜 힘들었겠다.",
            "그치? 나도 그런 적 있어.",
            "야, 너 진짜 고생 많았어.",
            "오늘은 좀 쉬어. 진심으로.",
            "힘내! 넌 잘하고 있어.",
        ]
    ),
    
    PersonaType.CALM_MENTOR: Persona(
        type=PersonaType.CALM_MENTOR,
        name="차분한 멘토",
        description="인생 선배처럼 담담하게 조언합니다. 큰 그림을 보여주며 안정감을 줍니다.",
        tone="담담하고 깊이 있는",
        style_guide=[
            "상황을 객관적으로 바라보도록 유도",
            "장기적 관점 제시",
            "지혜로운 격언이나 통찰 활용",
            "서두르지 않는 여유로운 말투",
            "'~하는 것도 하나의 방법이죠' 형태",
        ],
        emoji_usage="none",
        sample_phrases=[
            "이런 시간도 지나갈 거예요.",
            "한 걸음 물러서서 보면 다르게 보일 수 있어요.",
            "지금 이 순간이 전부는 아니에요.",
            "때로는 쉬어가는 것도 용기예요.",
            "당신은 이미 충분히 잘하고 있어요.",
        ]
    ),
    
    PersonaType.CHEERFUL_SUPPORTER: Persona(
        type=PersonaType.CHEERFUL_SUPPORTER,
        name="밝은 응원단",
        description="긍정적인 에너지로 응원합니다. 밝고 활기찬 톤으로 힘을 북돋아줍니다.",
        tone="밝고 에너지 넘치는",
        style_guide=[
            "긍정적 리프레이밍",
            "칭찬과 격려 중심",
            "감탄사 적극 활용",
            "짧고 힘있는 문장",
            "'~잖아요!', '~할 수 있어요!' 형태",
        ],
        emoji_usage="moderate",
        sample_phrases=[
            "와, 그래도 여기까지 온 거잖아요!",
            "대단해요! 정말 고생 많았어요!",
            "내일은 분명 더 좋아질 거예요!",
            "당신은 생각보다 강해요!",
            "이것도 다 경험이 될 거예요!",
        ]
    ),
}


# ============================================
# 카테고리별 컨텍스트
# ============================================

CATEGORY_CONTEXT = {
    "정서적_고갈": {
        "description": "에너지가 고갈되고 지친 상태입니다.",
        "core_feeling": "지침, 무기력, 피로",
        "user_needs": "휴식, 에너지 충전, 위로",
        "avoid": "더 노력하라는 말, 비교, 압박",
    },
    "좌절_압박": {
        "description": "억울하고 화가 나는 상태입니다.",
        "core_feeling": "분노, 억울함, 답답함",
        "user_needs": "감정 해소, 인정, 공정함",
        "avoid": "참으라는 말, 상대방 편들기, 감정 무시",
    },
    "부정적_대인관계": {
        "description": "관계에서 상처받은 상태입니다.",
        "core_feeling": "서운함, 외로움, 배신감",
        "user_needs": "공감, 지지, 소속감",
        "avoid": "혼자 있으라는 말, 관계 포기 유도, 비난",
    },
    "자기비하": {
        "description": "스스로를 탓하고 불안한 상태입니다.",
        "core_feeling": "자책, 불안, 열등감",
        "user_needs": "자존감 회복, 안심, 격려",
        "avoid": "비판, 비교, 실패 강조",
    },
    "긍정": {
        "description": "긍정적이고 안정된 상태입니다.",
        "core_feeling": "만족, 기쁨, 평온",
        "user_needs": "지지, 격려, 유지",
        "avoid": "과도한 칭찬, 경계심 유발",
    },
}


# ============================================
# 프롬프트 생성기
# ============================================

class PromptBuilder:
    """LLM 프롬프트 빌더"""
    
    def __init__(self, persona_type: PersonaType = PersonaType.WARM_COUNSELOR):
        self.persona = PERSONAS[persona_type]
    
    def set_persona(self, persona_type: PersonaType):
        """페르소나 변경"""
        self.persona = PERSONAS[persona_type]
    
    def build_feedback_prompt(
        self,
        category: str,
        user_text: str,
        keywords: List[str] = None,
        activity_name: str = None,
        history_summary: str = None,
    ) -> str:
        """피드백 생성 프롬프트 구성"""
        
        context = CATEGORY_CONTEXT.get(category, CATEGORY_CONTEXT["정서적_고갈"])
        
        # 스타일 가이드 문자열
        style_rules = "\n".join([f"- {rule}" for rule in self.persona.style_guide])
        
        # 샘플 문구
        sample_phrases = "\n".join([f'- "{phrase}"' for phrase in self.persona.sample_phrases[:3]])
        
        prompt = f"""### 역할
당신은 '{self.persona.name}'입니다.
{self.persona.description}

### 말투 스타일
톤: {self.persona.tone}
{style_rules}

### 참고할 말투 예시
{sample_phrases}

### 사용자 상태
- 감정 카테고리: {category}
- 상태 설명: {context['description']}
- 핵심 감정: {context['core_feeling']}
- 사용자가 필요로 하는 것: {context['user_needs']}
- 피해야 할 표현: {context['avoid']}

### 사용자 입력
일기 내용: "{user_text[:200] if user_text else '(직접 작성하지 않음)'}"
감정 키워드: {', '.join(keywords) if keywords else '없음'}
"""
        
        if activity_name:
            prompt += f"추천 활동: {activity_name}\n"
        
        if history_summary:
            prompt += f"최근 감정 흐름: {history_summary}\n"
        
        prompt += f"""
### 이모지 사용
{self.persona.emoji_usage} (none: 사용 안함, minimal: 최소, moderate: 적당히)

### 작성 규칙
1. 2-3문장으로 간결하게 작성
2. 감정을 먼저 인정하고 공감
3. {self.persona.tone} 톤 유지
4. 추천 활동이 있다면 자연스럽게 연결
5. 강요하지 않고 제안하는 방식

### 응답
"""
        return prompt
    
    def build_solution_prompt(
        self,
        category: str,
        activity_name: str,
        activity_description: str = None,
        user_keywords: List[str] = None,
    ) -> str:
        """솔루션 추천 메시지 프롬프트"""
        
        context = CATEGORY_CONTEXT.get(category, CATEGORY_CONTEXT["정서적_고갈"])
        
        prompt = f"""### 역할
당신은 '{self.persona.name}'입니다.
{self.persona.description}

### 목표
'{activity_name}' 활동을 {self.persona.tone} 톤으로 추천하는 한 문장을 작성하세요.

### 사용자 상태
- 감정: {category} ({context['core_feeling']})
- 키워드: {', '.join(user_keywords) if user_keywords else '없음'}

### 활동 정보
- 이름: {activity_name}
- 설명: {activity_description or '(없음)'}

### 작성 규칙
1. 한 문장으로 작성 (20-40자)
2. 활동을 자연스럽게 추천
3. 강요하지 않는 부드러운 제안
4. {self.persona.tone} 톤 유지

### 응답
"""
        return prompt


# ============================================
# 템플릿 기반 피드백 (LLM 없이)
# ============================================

FEEDBACK_TEMPLATES = {
    PersonaType.WARM_COUNSELOR: {
        "정서적_고갈": [
            "많이 지치셨네요. 오늘은 자신을 위한 시간을 가져보는 건 어떨까요?",
            "에너지가 바닥난 느낌이시죠. 잠깐 멈추고 숨 고르는 것도 용기예요.",
            "지친 마음, 충분히 이해해요. 오늘 하루 정말 수고 많으셨습니다.",
            "무기력한 느낌이 드시는군요. 아무것도 안 해도 괜찮아요.",
        ],
        "좌절_압박": [
            "억울하고 답답한 마음이 느껴져요. 그 감정은 당연한 거예요.",
            "화가 나는 건 자연스러운 감정이에요. 잠시 깊게 숨을 쉬어보세요.",
            "압박감 속에서도 버티고 계시네요. 정말 대단하세요.",
            "스트레스가 많으셨군요. 그 마음 충분히 이해해요.",
        ],
        "부정적_대인관계": [
            "관계에서 상처받으셨군요. 그 마음이 얼마나 힘드실지 느껴져요.",
            "사람 사이에서 오는 스트레스는 정말 힘들죠. 혼자가 아니에요.",
            "서운한 마음, 충분히 이해해요. 당신 잘못이 아니에요.",
            "힘든 관계 속에서 애쓰셨네요. 자신을 먼저 챙겨주세요.",
        ],
        "자기비하": [
            "자신을 너무 탓하지 마세요. 당신은 충분히 잘하고 있어요.",
            "불안한 마음이 드시는군요. 그래도 당신은 가치 있는 사람이에요.",
            "완벽하지 않아도 괜찮아요. 지금 이 순간도 잘 해내고 있어요.",
            "스스로를 너무 몰아세우지 마세요. 쉬어가도 괜찮습니다.",
        ],
        "긍정": [
            "오늘 하루도 수고하셨어요! 좋은 에너지가 느껴지네요.",
            "긍정적인 하루를 보내고 계시네요. 그 기운 계속 이어가세요.",
            "밝은 마음이 느껴져요. 오늘도 좋은 하루 되세요.",
        ],
    },
    
    PersonaType.PRACTICAL_ADVISOR: {
        "정서적_고갈": [
            "지쳤을 땐 무리하지 마세요. 오늘은 일찍 쉬는 게 좋겠어요.",
            "에너지 충전이 필요해 보여요. 10분이라도 눈을 감고 쉬어보세요.",
            "번아웃의 신호일 수 있어요. 당장 급한 일만 하고 나머지는 내일로 미뤄보세요.",
        ],
        "좌절_압박": [
            "화가 날 땐 잠시 자리를 피하는 것도 방법이에요.",
            "스트레스 상황에서 벗어나 5분만 바람 쐬고 오세요.",
            "지금 당장 해결 안 되는 건 잠시 내려놓아도 괜찮아요.",
        ],
        "부정적_대인관계": [
            "관계 문제는 시간이 필요해요. 오늘은 거리를 두는 게 좋겠어요.",
            "힘든 사람과는 당분간 접촉을 줄여보세요.",
            "내 감정부터 추스르는 게 먼저예요. 상대방은 그 다음이에요.",
        ],
        "자기비하": [
            "실수는 누구나 해요. 오늘 잘한 것 하나만 떠올려보세요.",
            "자책보다는 다음에 어떻게 할지 생각하는 게 도움이 돼요.",
            "불안할 땐 심호흡 3번. 지금 당장 해보세요.",
        ],
        "긍정": [
            "좋은 컨디션이네요. 이 기운으로 하고 싶던 일 하나 해보세요.",
            "긍정적인 에너지가 느껴져요. 오늘 하루도 잘 보내세요.",
        ],
    },
    
    PersonaType.FRIENDLY_BUDDY: {
        "정서적_고갈": [
            "에휴, 진짜 힘들었겠다. 오늘은 그냥 푹 쉬어!",
            "야, 너 요즘 너무 무리한 거 아니야? 좀 쉬어.",
            "지쳤구나... 오늘은 아무것도 하지 마. 그냥 쉬어.",
        ],
        "좌절_압박": [
            "헐, 진짜 열받았겠다. 나라도 화났을 듯.",
            "아 진짜? 그건 좀 너무하네. 화낼 만해.",
            "스트레스 장난 아니겠다. 맛있는 거 먹어!",
        ],
        "부정적_대인관계": [
            "에이, 그 사람이 이상한 거야. 네 잘못 아니야.",
            "사람 때문에 스트레스받는 거 진짜 힘들지...",
            "그치? 관계가 제일 피곤해. 힘내.",
        ],
        "자기비하": [
            "야, 너 뭔 소리야. 넌 잘하고 있어 진짜로.",
            "너무 자책하지 마. 누구나 실수해.",
            "야, 넌 충분히 괜찮은 사람이야. 진심이야.",
        ],
        "긍정": [
            "오 좋다! 오늘 기분 좋은 거야? ㅎㅎ",
            "굿굿~ 그 기운 계속 가져가자!",
        ],
    },
    
    PersonaType.CALM_MENTOR: {
        "정서적_고갈": [
            "지금 지친 건 그만큼 열심히 살아왔다는 증거예요.",
            "쉬어가는 것도 일의 일부입니다. 조급해하지 마세요.",
            "이런 시간도 지나갑니다. 지금은 잠시 멈춰도 괜찮아요.",
        ],
        "좌절_압박": [
            "화는 자연스러운 감정이에요. 다만 그것에 휩쓸리지 않으면 좋겠어요.",
            "한 걸음 물러서면 다르게 보일 수 있어요.",
            "지금 이 상황이 영원하진 않아요. 때를 기다려보세요.",
        ],
        "부정적_대인관계": [
            "모든 관계가 다 맞을 순 없어요. 거리를 두는 것도 지혜예요.",
            "사람 사이의 일은 시간이 해결해주기도 해요.",
            "당신의 가치는 타인의 평가로 정해지지 않아요.",
        ],
        "자기비하": [
            "완벽한 사람은 없어요. 부족함도 당신의 일부예요.",
            "실수는 성장의 과정이에요. 너무 자책하지 마세요.",
            "스스로를 너그럽게 대해주세요. 당신은 그럴 자격이 있어요.",
        ],
        "긍정": [
            "좋은 에너지가 느껴지네요. 그 평온함을 오래 유지하세요.",
            "균형 잡힌 하루를 보내고 계시군요. 잘하고 있어요.",
        ],
    },
    
    PersonaType.CHEERFUL_SUPPORTER: {
        "정서적_고갈": [
            "힘들었지만 여기까지 온 거잖아요! 대단해요!",
            "지쳐도 포기 안 하는 당신, 진짜 멋있어요!",
            "오늘 푹 쉬고 내일 다시 달려봐요! 할 수 있어요!",
        ],
        "좌절_압박": [
            "화나는 거 당연해요! 그래도 당신은 잘하고 있어요!",
            "스트레스 많았죠? 그래도 이겨낼 수 있어요! 힘내요!",
            "답답하죠? 하지만 당신은 충분히 강해요!",
        ],
        "부정적_대인관계": [
            "관계 때문에 힘들었죠? 하지만 당신 편도 분명 있어요!",
            "사람 때문에 지쳤어도, 당신은 여전히 멋진 사람이에요!",
            "힘든 관계는 정리해도 괜찮아요! 당신이 더 소중해요!",
        ],
        "자기비하": [
            "뭔 소리예요! 당신은 정말 잘하고 있어요!",
            "실수? 괜찮아요! 다음엔 더 잘할 수 있어요!",
            "자신감 가져요! 당신은 생각보다 훨씬 멋진 사람이에요!",
        ],
        "긍정": [
            "오늘 기분 좋은 거 맞죠?! 최고예요! 👏",
            "긍정 에너지 뿜뿜! 오늘도 좋은 하루 되세요! ✨",
        ],
    },
}


def get_template_feedback(
    persona_type: PersonaType,
    category: str,
    keywords: List[str] = None
) -> str:
    """템플릿 기반 피드백 반환"""
    import random
    
    templates = FEEDBACK_TEMPLATES.get(persona_type, FEEDBACK_TEMPLATES[PersonaType.WARM_COUNSELOR])
    category_templates = templates.get(category, templates.get("정서적_고갈", []))
    
    if not category_templates:
        return "오늘 하루도 수고하셨어요."
    
    feedback = random.choice(category_templates)
    
    # 키워드 연결 (따뜻한 상담사, 차분한 멘토만)
    if keywords and persona_type in [PersonaType.WARM_COUNSELOR, PersonaType.CALM_MENTOR]:
        keyword_str = ", ".join(keywords[:2])
        feedback += f" '{keyword_str}'라는 말씀에서 마음이 느껴졌어요."
    
    return feedback


# ============================================
# 페르소나 검증 패턴
# ============================================

PERSONA_VALIDATION: Dict[PersonaType, dict] = {
    PersonaType.WARM_COUNSELOR: {
        "required": ["요", "네요", "죠", "어요", "습니다"],
        "forbidden": ["야", "해", "지ㅋ", "ㅋㅋ"],
    },
    PersonaType.PRACTICAL_ADVISOR: {
        "required": ["요", "세요", "어요", "습니다", "보세요"],
        "forbidden": ["야", "ㅋㅋ", "에휴", "헐"],
    },
    PersonaType.FRIENDLY_BUDDY: {
        "required": ["야", "해", "지", "다", "어", "ㅎㅎ", "ㅋ"],
        "forbidden": ["습니다", "드립니다", "하세요", "드세요"],
    },
    PersonaType.CALM_MENTOR: {
        "required": ["요", "어요", "죠", "습니다", "네요"],
        "forbidden": ["야", "ㅋㅋ", "헐", "에휴", "!"],
    },
    PersonaType.CHEERFUL_SUPPORTER: {
        "required": ["요", "어요", "!"],
        "forbidden": ["야", "ㅋㅋ", "습니다"],
    },
}


# ============================================
# 유틸리티 함수
# ============================================

def get_persona_by_preference(preference: dict) -> PersonaType:
    """
    사용자 설문 결과(preference)에 따라 적절한 페르소나 반환
    
    preference 예시:
    {
        "is_active": True,      # 활동적인지
        "is_social": True,      # 사교적인지
        "preferred_tone": "warm"  # 선호 톤 (있다면)
    }
    """
    # 선호 톤이 명시되어 있으면 그대로 사용
    tone_mapping = {
        "warm": PersonaType.WARM_COUNSELOR,
        "practical": PersonaType.PRACTICAL_ADVISOR,
        "friendly": PersonaType.FRIENDLY_BUDDY,
        "calm": PersonaType.CALM_MENTOR,
        "cheerful": PersonaType.CHEERFUL_SUPPORTER,
    }
    
    if preferred_tone := preference.get("preferred_tone"):
        if preferred_tone in tone_mapping:
            return tone_mapping[preferred_tone]
    
    # 선호 톤이 없으면 성향에 따라 추천
    is_active = preference.get("is_active", False)
    is_social = preference.get("is_social", False)
    
    if is_active and is_social:
        return PersonaType.CHEERFUL_SUPPORTER
    elif is_active and not is_social:
        return PersonaType.PRACTICAL_ADVISOR
    elif not is_active and is_social:
        return PersonaType.FRIENDLY_BUDDY
    else:
        return PersonaType.WARM_COUNSELOR


def list_personas() -> List[dict]:
    """모든 페르소나 목록 반환"""
    return [
        {
            "type": p.type.value,
            "name": p.name,
            "description": p.description,
            "tone": p.tone,
        }
        for p in PERSONAS.values()
    ]


# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    # 프롬프트 빌더 테스트
    builder = PromptBuilder(PersonaType.WARM_COUNSELOR)
    
    prompt = builder.build_feedback_prompt(
        category="정서적_고갈",
        user_text="오늘 회사에서 야근하고 집에 오니까 12시였어. 너무 지친다.",
        keywords=["지침", "피곤"],
        activity_name="따뜻한 차 마시기"
    )
    
    print("=" * 60)
    print("🎭 프롬프트 예시 (따뜻한 상담사)")
    print("=" * 60)
    print(prompt)
    
    print("\n" + "=" * 60)
    print("📝 템플릿 피드백 예시")
    print("=" * 60)
    
    for persona_type in PersonaType:
        feedback = get_template_feedback(
            persona_type=persona_type,
            category="좌절_압박",
            keywords=["스트레스", "화남"]
        )
        print(f"\n[{PERSONAS[persona_type].name}]")
        print(f"  → {feedback}")
