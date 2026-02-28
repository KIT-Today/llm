# -*- coding: utf-8 -*-
"""
번아웃 분석 엔진
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from config import Config
from constants import STAGE1_CATEGORIES, STAGE2_CATEGORIES, MBI_CATEGORY_MAP, BURNOUT_KEYWORDS


class BurnoutClassifier(nn.Module):
    """2단계 분류 모델"""
    
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


class BurnoutAnalyzer:
    """번아웃 분석기 (싱글톤)"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """모델 로드"""
        if self._initialized:
            return
        
        print(f"모델 로딩 중... (Device: {Config.DEVICE})")
        
        # KURE 임베딩 모델
        self.kure = SentenceTransformer("nlpai-lab/KURE-v1", device=Config.DEVICE)
        
        # Stage 1: 긍정/부정 분류
        s1_path = f"{Config.MODEL_DIR}/stage1_model.pt"
        s1_ckpt = torch.load(s1_path, map_location=Config.DEVICE, weights_only=False)
        self.stage1 = BurnoutClassifier(
            input_dim=s1_ckpt.get('embedding_dim', 1024),
            hidden_dim=s1_ckpt.get('hidden_dim', 256),
            num_classes=2
        ).to(Config.DEVICE)
        self.stage1.load_state_dict(s1_ckpt['model_state_dict'])
        self.stage1.eval()
        
        # Stage 2: 4가지 번아웃 카테고리 분류
        s2_path = f"{Config.MODEL_DIR}/stage2_model.pt"
        s2_ckpt = torch.load(s2_path, map_location=Config.DEVICE, weights_only=False)
        self.stage2 = BurnoutClassifier(
            input_dim=s2_ckpt.get('embedding_dim', 1024),
            hidden_dim=s2_ckpt.get('hidden_dim', 256),
            num_classes=4
        ).to(Config.DEVICE)
        self.stage2.load_state_dict(s2_ckpt['model_state_dict'])
        self.stage2.eval()
        
        self._initialized = True
        print("모델 로딩 완료!")
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """텍스트 임베딩 생성"""
        return self.kure.encode(text, convert_to_tensor=True).unsqueeze(0).to(Config.DEVICE)
    
    def predict_stage1(self, text: str) -> tuple:
        """1단계: 긍정/부정 분류"""
        with torch.no_grad():
            emb = self._get_embedding(text)
            logits = self.stage1(emb)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return pred, probs
    
    def predict_stage2(self, text: str) -> tuple:
        """2단계: 번아웃 카테고리 분류"""
        with torch.no_grad():
            emb = self._get_embedding(text)
            logits = self.stage2(emb)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return pred, probs
    
    def extract_keywords(self, text: str, category: str, top_k: int = 3) -> List[str]:
        """키워드 추출"""
        if category not in BURNOUT_KEYWORDS:
            return []
        keywords = BURNOUT_KEYWORDS[category]["keywords"]
        matched = [kw for kw in keywords if kw in text]
        return matched[:top_k]
    
    def analyze(self, text: str, keywords: Optional[Dict] = None) -> Dict:
        """전체 분석 파이프라인"""
        # 텍스트 준비
        analysis_text = text or ""
        if keywords:
            keyword_text = " ".join([f"{k}: {v}" if isinstance(v, str) else str(v) for k, v in keywords.items()])
            analysis_text = f"{analysis_text} {keyword_text}".strip()
        
        # 빈 텍스트 처리
        if not analysis_text:
            return {
                "primary_emotion": "긍정",
                "primary_score": 0.5,
                "mbi_category": "NORMAL",
                "emotion_probs": {"긍정": 0.5, "부정": 0.5},
                "burnout_category": None,
                "keywords": []
            }

        # Stage 1: 긍정/부정
        s1_pred, s1_probs = self.predict_stage1(analysis_text)
        primary_emotion = STAGE1_CATEGORIES[s1_pred]

        # Stage 2 카테고리 확률 기본값 -1.0 (긍정이면 Stage 2 미실행 → 0.0과 구분)
        s2_probs_map = {cat: -1.0 for cat in STAGE2_CATEGORIES.values()}

        result = {
            "primary_emotion": primary_emotion,
            "primary_score": float(s1_probs[s1_pred]),
            "emotion_probs": {"긍정": float(s1_probs[0]), "부정": float(s1_probs[1]), **s2_probs_map},
            "burnout_category": None,
            "mbi_category": "NORMAL",   # Stage 1 긍정 → NORMAL, Stage 2 부정 → 한국어 카테고리
            "keywords": []
        }

        # Stage 2: 번아웃 카테고리 (부정인 경우만)
        if s1_pred == 1:
            s2_pred, s2_probs = self.predict_stage2(analysis_text)
            burnout_category = STAGE2_CATEGORIES[s2_pred]
            result["burnout_category"] = burnout_category
            result["mbi_category"] = burnout_category  # 한국어 카테고리 그대로 전송 (백엔드가 EE/DP/PA로 변환)
            result["keywords"] = self.extract_keywords(analysis_text, burnout_category)
            # Stage 2 카테고리별 확률로 덮어쓰기
            for i, cat in STAGE2_CATEGORIES.items():
                result["emotion_probs"][cat] = round(float(s2_probs[i]), 4)
        
        return result
