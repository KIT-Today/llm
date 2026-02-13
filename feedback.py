# -*- coding: utf-8 -*-
"""
피드백 생성기
"""

import torch
from typing import List

from prompts import (
    PersonaType,
    PERSONAS,
    PromptBuilder,
    get_template_feedback,
)


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
        
        # 활동 추천 메시지 추가
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
        """LLM 기반 피드백"""
        persona = PERSONAS[self.persona_type]
        
        activity_instruction = ""
        if activity_name:
            activity_instruction = f"\n- '\"{activity_name}\"' 활동을 자연스럽게 추천"
        
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
            
            if "\n\n" in response:
                response = response.split("\n\n")[0].strip()
            
            if len(response) < 10:
                return self._generate_template(category, keywords)
            
            return response
            
        except Exception as e:
            print(f"LLM 생성 실패: {e}")
            return self._generate_template(category, keywords)
