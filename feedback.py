# -*- coding: utf-8 -*-
"""
피드백 생성기
"""

import re
import torch
from typing import List, Optional

from prompts import (
    PersonaType,
    PERSONAS,
    PERSONA_VALIDATION,
    CATEGORY_CONTEXT,
    PromptBuilder,
    get_template_feedback,
)

# 피드백 길이 제한
MIN_FEEDBACK_LEN = 10
MAX_FEEDBACK_LEN = 150


class FeedbackGenerator:
    """페르소나 기반 피드백 생성"""

    def __init__(self, use_llm: bool = False, persona_type: PersonaType = PersonaType.WARM_COUNSELOR):
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

    # --------------------------------------------------
    # 후처리 & 검증
    # --------------------------------------------------

    def _postprocess(self, text: str) -> Optional[str]:
        """
        LLM 출력 후처리.
        문제가 있으면 None 반환 → 템플릿 fallback 트리거.

        검사 순서:
        1. 프롬프트 지시어 누출 제거
        2. 이모지 제거
        3. \\n\\n 이후 잘라내기
        4. 길이 범위 검사 (MIN ~ MAX)
        5. 불완전 문장 감지
        """
        # 1. 프롬프트 지시어 누출 제거
        for marker in ["### 응답", "### 명령어", "규칙:", "입력:"]:
            if marker in text:
                text = text.split(marker)[-1]

        text = text.strip()

        # 2. 이모지 제거
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

        # 3. \n\n 이후 잘라내기 (다른 주제로 이어지는 경우)
        if "\n\n" in text:
            text = text.split("\n\n")[0].strip()

        # 4. 길이 범위 검사
        if len(text) < MIN_FEEDBACK_LEN:
            return None
        if len(text) > MAX_FEEDBACK_LEN:
            # 마지막 완전한 문장까지만 자르기
            sentences = re.split(r'(?<=[.!?요다죠어])\s+', text)
            truncated = ""
            for s in sentences:
                if len(truncated) + len(s) <= MAX_FEEDBACK_LEN:
                    truncated += s + " "
                else:
                    break
            text = truncated.strip()
            # 잘랐는데도 너무 짧으면 버림
            if len(text) < MIN_FEEDBACK_LEN:
                return None

        # 5. 불완전 문장 감지 (요/다/죠/어/!/? 로 끝나야 함)
        if not re.search(r'[.!?\uC694\uB2E4\uC8E0\uC5B4]\s*$', text):
            return None

        return text

    def _check_repetition(self, text: str) -> bool:
        """
        반복 표현 감지.
        문장 내 4-gram 겹침이 50% 이상이면 False 반환.
        """
        words = text.split()
        if len(words) < 8:
            return True  # 짧은 문장은 검사 생략

        ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        if not ngrams:
            return True

        unique_ratio = len(set(ngrams)) / len(ngrams)
        return unique_ratio >= 0.5  # 50% 이상 고유해야 통과

    def _validate_persona(self, text: str) -> bool:
        """
        페르소나 톤 일관성 검사.
        required 패턴 하나라도 포함 AND forbidden 패턴 없어야 통과.
        """
        rules = PERSONA_VALIDATION.get(self.persona_type)
        if not rules:
            return True

        has_required = any(p in text for p in rules["required"])
        has_forbidden = any(p in text for p in rules["forbidden"])

        return has_required and not has_forbidden

    def _check_category_relevance(self, text: str, category: str) -> bool:
        """
        카테고리 키워드 관련성 검사.
        CATEGORY_CONTEXT의 core_feeling 키워드 중 하나라도 포함되면 통과.
        없으면 통과시키되 경고 로깅 (너무 엄격하면 fallback 과다 발생).
        """
        context = CATEGORY_CONTEXT.get(category)
        if not context:
            return True

        # core_feeling 쉼표 구분 키워드 추출
        keywords = [k.strip() for k in context["core_feeling"].split(",")]
        # avoid 표현이 피드백에 포함되면 탈락
        avoid_keywords = [k.strip() for k in context["avoid"].split(",")]

        has_avoid = any(k in text for k in avoid_keywords if len(k) > 1)
        if has_avoid:
            print(f"카테고리 관련성 경고: avoid 키워드 감지 ({category})")
            return False

        return True

    def _inject_keyword(self, text: str, keywords: List[str]) -> str:
        """
        사용자 키워드 반영 확인 및 자연스러운 삽입.
        키워드가 이미 포함돼 있으면 그대로 반환.
        없으면 첫 번째 키워드를 첫 문장 끝에 자연스럽게 삽입.
        """
        if not keywords:
            return text

        top_keyword = keywords[0]

        # 이미 반영돼 있으면 패스
        if top_keyword in text:
            return text

        # 첫 문장 끝에 삽입 (마침표/요/다 앞)
        injected = re.sub(
            r'([.!?요다죠어])(\s|$)',
            lambda m: f"({top_keyword})" + m.group(1) + m.group(2),
            text,
            count=1
        )
        return injected

    # --------------------------------------------------
    # 외부 인터페이스
    # --------------------------------------------------

    def generate(self, category: str, user_text: str = "", keywords: List[str] = None, activity_name: str = "") -> str:
        """피드백 생성"""
        if self.use_llm and self.generator:
            return self._generate_llm(category, user_text, keywords, activity_name)
        return self._generate_template(category, keywords, activity_name)

    def _generate_template(self, category: str, keywords: List[str] = None, activity_name: str = "") -> str:
        """템플릿 기반 피드백"""
        feedback = get_template_feedback(
            persona_type=self.persona_type,
            category=category,
            keywords=keywords
        )

        if activity_name:
            if self.persona_type == PersonaType.FRIENDLY_BUDDY:
                feedback += f" '{activity_name}' 어때?"
            elif self.persona_type == PersonaType.CHEERFUL_SUPPORTER:
                feedback += f" '{activity_name}' 한번 해봐요!"
            elif self.persona_type == PersonaType.PRACTICAL_ADVISOR:
                feedback += f" '{activity_name}'을(를) 추천드려요."
            else:
                feedback += f" '{activity_name}'은(는) 어떨까요?"

        return feedback

    def _generate_llm(self, category: str, user_text: str, keywords: List[str], activity_name: str = "") -> str:
        """
        LLM 기반 피드백.
        후처리 파이프라인:
          1. _postprocess()         — 누출/이모지/길이/불완전 문장
          2. _check_repetition()    — 반복 표현 감지
          3. _validate_persona()    — 페르소나 톤 패턴
          4. _check_category_relevance() — 카테고리 관련성
          5. _inject_keyword()      — 사용자 키워드 반영 (통과 후 적용)
        2회 재시도 후 모두 실패 시 템플릿 fallback.
        """
        persona = PERSONAS[self.persona_type]

        activity_instruction = ""
        if activity_name:
            activity_instruction = f"\n- '{activity_name}' 활동을 자연스럽게 추천"

        prompt = f"""### 명령어:
당신은 '{persona.name}'입니다. {persona.description}
번아웃을 겪는 직장인에게 {persona.tone} 톤으로 2-3문장의 공감 메시지를 작성하세요.

규칙:
- {persona.tone} 톤 유지
- 감정을 인정하고 공감
- 강요하지 않고 부드럽게 제안
- 이모지 사용 금지{activity_instruction}

### 입력:
감정 상태: {category}
사용자 일기: "{user_text[:150] if user_text else '(내용 없음)'}"
주요 키워드: {', '.join(keywords) if keywords else '없음'}
추천 활동: {activity_name if activity_name else '없음'}

### 응답:
"""

        for attempt in range(2):
            try:
                result = self.generator(
                    prompt,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7 + attempt * 0.1,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                raw = result[0]['generated_text']
                response = raw.split("### 응답:")[-1].strip()

                # 1. 기본 후처리
                response = self._postprocess(response)
                if response is None:
                    print(f"[{attempt+1}/2] 후처리 실패")
                    continue

                # 2. 반복 표현 감지
                if not self._check_repetition(response):
                    print(f"[{attempt+1}/2] 반복 표현 감지")
                    continue

                # 3. 페르소나 톤 검증
                if not self._validate_persona(response):
                    print(f"[{attempt+1}/2] 페르소나 불일치")
                    continue

                # 4. 카테고리 관련성 검사
                if not self._check_category_relevance(response, category):
                    print(f"[{attempt+1}/2] 카테고리 관련성 실패")
                    continue

                # 5. 사용자 키워드 반영 (통과 후 적용)
                response = self._inject_keyword(response, keywords or [])

                return response

            except Exception as e:
                print(f"[{attempt+1}/2] LLM 생성 실패: {e}")

        # 2회 모두 실패 → 템플릿 fallback
        print("템플릿 fallback 사용")
        return self._generate_template(category, keywords, activity_name)
