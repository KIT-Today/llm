
"""
🔍 번아웃 2단계 분류 판단 근거 분석 모듈
=====================================

3가지 방법으로 예측 근거를 설명:
1. Attention Score - 어떤 단어가 중요했는지
2. SHAP - 각 단어의 기여도 수치화
3. 키워드 매칭 - 사전 정의 키워드 매칭률
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

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
