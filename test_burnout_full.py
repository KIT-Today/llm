"""
🔥 번아웃 2단계 분류 + 피드백 생성 테스트
==========================================

사용법:
    python test_burnout_full.py                    # 샘플 테스트
    python test_burnout_full.py --text "힘들다"    # 단일 텍스트
    python test_burnout_full.py -i                 # 인터랙티브 모드

필요 파일:
    - stage1_model.pt
    - stage2_model.pt

필요 패키지:
    pip install torch sentence-transformers transformers accelerate
"""

import argparse
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================
# 설정
# ============================================

STAGE1_CATEGORIES = {0: "긍정", 1: "부정"}
STAGE2_CATEGORIES = {0: "정서적_고갈", 1: "좌절_압박", 2: "부정적_대인관계", 3: "자기비하"}

BURNOUT_KEYWORDS = {
    "긍정": {"keywords": ["좋다", "좋아", "행복", "기쁘", "뿌듯", "만족", "감사", "고맙", "다행", "홀가분", "상쾌", "힐링", "편안", "여유", "성공", "달성", "완료", "끝났", "칭찬", "인정", "보람", "즐겁", "신나", "설레", "기대", "희망", "웃"]},
    "부정": {"keywords": ["힘들", "지치", "피곤", "싫", "짜증", "화나", "억울", "슬프", "우울", "불안", "걱정", "무섭", "두렵", "외롭", "서운", "실망", "후회", "미안"]},
    "정서적_고갈": {"keywords": ["지치", "피곤", "힘들", "무기력", "탈진", "녹초", "방전", "지쳐", "의욕", "에너지", "기운", "무거", "공허", "텅", "비어", "메마르", "번아웃", "우울", "슬프", "눈물", "허무", "무의미", "싫어", "귀찮"]},
    "좌절_압박": {"keywords": ["화나", "화가", "짜증", "열받", "빡치", "분노", "억울", "불공평", "압박", "스트레스", "마감", "답답", "미치", "폭발", "한계", "못참", "왜", "도대체", "짓눌", "감당", "부담", "실적", "안되", "안풀"]},
    "부정적_대인관계": {"keywords": ["무시", "소외", "따돌", "왕따", "배신", "뒷담", "험담", "갈등", "싸우", "다투", "틀어", "소문", "오해", "믿었", "실망", "서운", "혼자", "외로", "편없", "거절", "빼고", "안끼", "정치", "눈치"]},
    "자기비하": {"keywords": ["못하", "못난", "부족", "무능", "한심", "자격", "불안", "걱정", "자책", "죄책", "잘못", "내탓", "미안", "후회", "열등", "비교", "왜나만", "자신없", "두렵", "무섭", "실패", "망", "가치없", "쓸모없"]},
}

# 피드백 템플릿 (LLM 없이 사용)
FEEDBACK_TEMPLATES = {
    "긍정": [
        "오늘 하루도 수고하셨어요! 좋은 에너지가 느껴지네요. 😊",
        "긍정적인 하루를 보내고 계시네요. 그 기운 계속 이어가세요!",
    ],
    "정서적_고갈": [
        "많이 지치셨네요. 오늘은 일찍 쉬어보는 건 어떨까요?",
        "에너지가 바닥난 느낌이시죠. 잠깐 숨 고르는 시간이 필요해요.",
        "지친 마음, 충분히 이해해요. 오늘 하루 정말 수고 많으셨어요.",
    ],
    "좌절_압박": [
        "억울하고 답답한 마음이 느껴져요. 그 감정은 당연한 거예요.",
        "화가 나는 건 자연스러운 감정이에요. 잠시 깊게 숨을 쉬어보세요.",
        "압박감 속에서도 버티고 계시네요. 대단하세요. 잠시 쉬어가도 괜찮아요.",
    ],
    "부정적_대인관계": [
        "관계에서 상처받으셨군요. 그 마음이 얼마나 힘드실지 느껴져요.",
        "사람 사이에서 오는 스트레스는 정말 힘들죠. 혼자가 아니에요.",
        "서운한 마음, 충분히 이해해요. 당신 잘못이 아니에요.",
    ],
    "자기비하": [
        "자신을 너무 탓하지 마세요. 당신은 충분히 잘하고 있어요.",
        "불안한 마음이 드시는군요. 그래도 당신은 가치 있는 사람이에요.",
        "완벽하지 않아도 괜찮아요. 지금 이 순간도 잘 해내고 있어요.",
    ],
}


# ============================================
# 모델 정의
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
# Explainer
# ============================================

class BurnoutExplainer:
    def __init__(self, kure_model, stage1_model, stage2_model, device="cpu"):
        self.kure = kure_model
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.device = device

    def _tokenize(self, text):
        tokens = re.findall(r"[가-힣]+", text)
        return [t for t in tokens if len(t) >= 2]

    def _predict_stage1(self, text):
        self.stage1.eval()
        with torch.no_grad():
            emb = self.kure.encode(text, convert_to_tensor=True).unsqueeze(0).to(self.device)
            logits = self.stage1(emb)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return pred, probs

    def _predict_stage2(self, text):
        self.stage2.eval()
        with torch.no_grad():
            emb = self.kure.encode(text, convert_to_tensor=True).unsqueeze(0).to(self.device)
            logits = self.stage2(emb)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return pred, probs

    def attention_analysis(self, text, stage="stage1", top_k=5):
        tokens = self._tokenize(text)
        if not tokens:
            return []
        if stage == "stage1":
            base_pred, base_probs = self._predict_stage1(text)
        else:
            base_pred, base_probs = self._predict_stage2(text)
        base_conf = base_probs[base_pred]
        importance = {}
        for token in tokens:
            modified = text.replace(token, "", 1)
            if modified.strip():
                if stage == "stage1":
                    _, new_probs = self._predict_stage1(modified)
                else:
                    _, new_probs = self._predict_stage2(modified)
                change = base_conf - new_probs[base_pred]
                importance[token] = max(0, change)
        total = sum(importance.values()) + 1e-10
        importance = {k: v/total for k, v in importance.items()}
        sorted_tokens = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"token": t, "score": round(s, 4)} for t, s in sorted_tokens]

    def shap_analysis(self, text, stage="stage1", top_k=5):
        tokens = self._tokenize(text)
        if not tokens:
            return []
        if stage == "stage1":
            base_pred, base_probs = self._predict_stage1(text)
        else:
            base_pred, base_probs = self._predict_stage2(text)
        contributions = {}
        for token in tokens:
            modified = text.replace(token, "", 1)
            if modified.strip():
                if stage == "stage1":
                    _, new_probs = self._predict_stage1(modified)
                else:
                    _, new_probs = self._predict_stage2(modified)
                contrib = base_probs[base_pred] - new_probs[base_pred]
                contributions[token] = contrib
        sorted_tokens = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        return [{"token": t, "contribution": round(c, 4), "direction": "positive" if c > 0 else "negative"} for t, c in sorted_tokens]

    def keyword_analysis(self, text, categories=None):
        if categories is None:
            categories = list(BURNOUT_KEYWORDS.keys())
        text_lower = text.lower()
        results = {}
        for cat in categories:
            if cat not in BURNOUT_KEYWORDS:
                continue
            keywords = BURNOUT_KEYWORDS[cat]["keywords"]
            matched = [kw for kw in keywords if kw in text_lower]
            match_rate = len(matched) / len(keywords) if keywords else 0
            results[cat] = {"matched_keywords": matched, "match_count": len(matched), "match_rate": round(match_rate, 4)}
        return results

    def explain(self, text, top_k=5):
        s1_pred, s1_probs = self._predict_stage1(text)
        result = {
            "text": text,
            "stage1": {
                "prediction": {"label": s1_pred, "category": STAGE1_CATEGORIES[s1_pred], "confidence": float(s1_probs[s1_pred])},
                "probabilities": {STAGE1_CATEGORIES[i]: float(p) for i, p in enumerate(s1_probs)},
                "attention": self.attention_analysis(text, "stage1", top_k),
                "shap": self.shap_analysis(text, "stage1", top_k),
                "keywords": self.keyword_analysis(text, ["긍정", "부정"])
            }
        }
        if s1_pred == 1:
            s2_pred, s2_probs = self._predict_stage2(text)
            result["stage2"] = {
                "prediction": {"label": s2_pred, "category": STAGE2_CATEGORIES[s2_pred], "confidence": float(s2_probs[s2_pred])},
                "probabilities": {STAGE2_CATEGORIES[i]: float(p) for i, p in enumerate(s2_probs)},
                "attention": self.attention_analysis(text, "stage2", top_k),
                "shap": self.shap_analysis(text, "stage2", top_k),
                "keywords": self.keyword_analysis(text, list(STAGE2_CATEGORIES.values()))
            }
        return result


# ============================================
# 피드백 생성
# ============================================

def generate_feedback_template(category, keywords=None):
    """템플릿 기반 피드백 생성"""
    import random
    templates = FEEDBACK_TEMPLATES.get(category, FEEDBACK_TEMPLATES["정서적_고갈"])
    return random.choice(templates)


def generate_feedback_llm(category, user_text, keywords, generator, tokenizer):
    """LLM 기반 피드백 생성"""
    CATEGORY_CONTEXT = {
        "정서적_고갈": "지치고 무기력한 상태입니다.",
        "좌절_압박": "억울하고 화가 나는 상태입니다.",
        "부정적_대인관계": "대인관계에서 상처받은 상태입니다.",
        "자기비하": "자신을 탓하고 불안해하는 상태입니다."
    }
    
    prompt = f"""### 명령어:
당신은 직장인의 번아웃을 케어하는 따뜻한 상담사입니다.
사용자의 감정 상태를 보고, 공감하며 위로하는 2-3문장을 작성하세요.

규칙:
- 따뜻하고 부드러운 톤
- 사용자의 감정을 인정하고 공감
- 작은 행동 제안 (강요하지 않기)

### 입력:
감정 상태: {category} - {CATEGORY_CONTEXT.get(category, '')}
사용자 일기: "{user_text}"
주요 키워드: {', '.join(keywords) if keywords else '없음'}

### 응답:
"""
    
    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated = result[0]['generated_text']
    response = generated.split("### 응답:")[-1].strip()
    response = response.split("\n\n")[0].strip()
    
    return response


# ============================================
# 전체 파이프라인
# ============================================

class BurnoutPipeline:
    def __init__(self, model_dir=".", device=None, use_llm=False):
        # 디바이스 설정
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🖥️ Device: {self.device}")
        
        # KURE 로드
        print("📥 Loading KURE...")
        from sentence_transformers import SentenceTransformer
        self.kure = SentenceTransformer("nlpai-lab/KURE-v1", device=self.device)
        
        # Stage 1 모델 로드
        print("📥 Loading Stage 1 model...")
        s1_path = f"{model_dir}/stage1_model.pt"
        s1_ckpt = torch.load(s1_path, map_location=self.device, weights_only=False)
        self.stage1 = BurnoutClassifier(
            input_dim=s1_ckpt.get('embedding_dim', 1024),
            hidden_dim=s1_ckpt.get('hidden_dim', 256),
            num_classes=2
        ).to(self.device)
        self.stage1.load_state_dict(s1_ckpt['model_state_dict'])
        self.stage1.eval()
        
        # Stage 2 모델 로드
        print("📥 Loading Stage 2 model...")
        s2_path = f"{model_dir}/stage2_model.pt"
        s2_ckpt = torch.load(s2_path, map_location=self.device, weights_only=False)
        self.stage2 = BurnoutClassifier(
            input_dim=s2_ckpt.get('embedding_dim', 1024),
            hidden_dim=s2_ckpt.get('hidden_dim', 256),
            num_classes=4
        ).to(self.device)
        self.stage2.load_state_dict(s2_ckpt['model_state_dict'])
        self.stage2.eval()
        
        # Explainer 생성
        self.explainer = BurnoutExplainer(self.kure, self.stage1, self.stage2, self.device)
        
        # LLM 로드 (선택)
        self.use_llm = use_llm
        self.generator = None
        self.tokenizer = None
        
        if use_llm:
            print("📥 Loading KoAlpaca...")
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
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
        
        print("✅ 모델 로드 완료!")
    
    def analyze(self, text):
        """전체 분석 실행"""
        # 분류 + 판단 근거
        explanation = self.explainer.explain(text)
        
        s1_result = explanation['stage1']['prediction']
        
        result = {
            'text': text,
            'stage1': s1_result,
            'is_positive': s1_result['category'] == '긍정',
            'explanation': explanation
        }
        
        # 부정이면 Stage 2 + 피드백
        if not result['is_positive']:
            s2_result = explanation['stage2']['prediction']
            result['stage2'] = s2_result
            
            # 키워드 추출
            keywords = [item['token'] for item in explanation['stage2']['attention'][:3]]
            result['keywords'] = keywords
            
            # 피드백 생성
            if self.use_llm and self.generator:
                feedback = generate_feedback_llm(
                    s2_result['category'], text, keywords,
                    self.generator, self.tokenizer
                )
            else:
                feedback = generate_feedback_template(s2_result['category'], keywords)
            result['feedback'] = feedback
        else:
            result['feedback'] = generate_feedback_template('긍정')
        
        return result
    
    def to_api_response(self, result):
        """API용 JSON 변환"""
        response = {
            "input_text": result['text'],
            "is_positive": result['is_positive'],
            "stage1": {
                "category": result['stage1']['category'],
                "confidence": round(result['stage1']['confidence'], 4)
            },
            "feedback": result['feedback']
        }
        
        if not result['is_positive']:
            response["stage2"] = {
                "category": result['stage2']['category'],
                "confidence": round(result['stage2']['confidence'], 4)
            }
            response["keywords"] = result['keywords']
            
            exp = result['explanation']['stage2']
            response["explanation"] = {
                "attention": exp['attention'][:3],
                "shap": exp['shap'][:3],
                "matched_keywords": exp['keywords'][result['stage2']['category']]['matched_keywords']
            }
        
        return response


# ============================================
# 출력 함수
# ============================================

def print_result(result):
    """결과 출력"""
    print("\n" + "="*70)
    print(f"📝 입력: {result['text']}")
    print("="*70)
    
    s1 = result['stage1']
    print(f"\n[Stage 1] {s1['category']} ({s1['confidence']:.1%})")
    
    if result['is_positive']:
        print("\n✅ 상태: 긍정")
    else:
        s2 = result['stage2']
        print(f"[Stage 2] {s2['category']} ({s2['confidence']:.1%})")
        print(f"\n⚠️ 상태: {s2['category']} (번아웃 징후)")
        print(f"🔑 키워드: {', '.join(result['keywords'])}")
        
        # 판단 근거
        exp = result['explanation']['stage2']
        print("\n📊 판단 근거:")
        print("  [Attention]", end=" ")
        for item in exp['attention'][:3]:
            print(f"{item['token']}({item['score']:.0%})", end=" ")
        print()
        print("  [SHAP]", end=" ")
        for item in exp['shap'][:3]:
            sign = "+" if item['direction'] == 'positive' else "-"
            print(f"{item['token']}({sign}{abs(item['contribution']):.3f})", end=" ")
        print()
    
    print(f"\n💬 피드백:\n{result['feedback']}")
    print("="*70)


# ============================================
# 메인
# ============================================

def main():
    parser = argparse.ArgumentParser(description="번아웃 2단계 분류 테스트")
    parser.add_argument("--text", "-t", type=str, help="분석할 텍스트")
    parser.add_argument("--interactive", "-i", action="store_true", help="인터랙티브 모드")
    parser.add_argument("--model-dir", "-m", type=str, default=".", help="모델 디렉토리")
    parser.add_argument("--use-llm", action="store_true", help="LLM 피드백 사용 (KoAlpaca)")
    parser.add_argument("--json", "-j", action="store_true", help="JSON 출력")
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = BurnoutPipeline(model_dir=args.model_dir, use_llm=args.use_llm)
    
    if args.text:
        # 단일 텍스트 분석
        result = pipeline.analyze(args.text)
        if args.json:
            print(json.dumps(pipeline.to_api_response(result), ensure_ascii=False, indent=2))
        else:
            print_result(result)
    
    elif args.interactive:
        # 인터랙티브 모드
        print("\n🎮 인터랙티브 모드 (종료: q)")
        print("-"*50)
        while True:
            text = input("\n입력> ").strip()
            if text.lower() in ['q', 'quit', 'exit', '종료']:
                print("👋 종료합니다.")
                break
            if not text:
                continue
            
            result = pipeline.analyze(text)
            if args.json:
                print(json.dumps(pipeline.to_api_response(result), ensure_ascii=False, indent=2))
            else:
                print_result(result)
    
    else:
        # 샘플 테스트
        test_texts = [
            "오늘 정말 최악이다.",
        ]
        
        print("\n🧪 샘플 테스트")
        for text in test_texts:
            result = pipeline.analyze(text)
            print_result(result)
        
        print("\n📤 API Response 예시:")
        result = pipeline.analyze("상사가 또 화를 냈다. 억울하고 분하다.")
        print(json.dumps(pipeline.to_api_response(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
