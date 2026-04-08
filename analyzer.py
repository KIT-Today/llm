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
        try:
            self.kure = SentenceTransformer("nlpai-lab/KURE-v1", device=Config.DEVICE)
        except Exception as e:
            raise RuntimeError(f"KURE 임베딩 모델 로드 실패: {e}")

        # Stage 1: 긍정/부정 분류
        s1_path = f"{Config.MODEL_DIR}/stage1_model.pt"
        try:
            s1_ckpt = torch.load(s1_path, map_location=Config.DEVICE, weights_only=False)
            self.stage1 = BurnoutClassifier(
                input_dim=s1_ckpt.get('embedding_dim', 1024),
                hidden_dim=s1_ckpt.get('hidden_dim', 256),
                num_classes=2
            ).to(Config.DEVICE)
            self.stage1.load_state_dict(s1_ckpt['model_state_dict'])
            self.stage1.eval()
        except FileNotFoundError:
            raise RuntimeError(f"Stage 1 모델 파일 없음: {s1_path}")
        except Exception as e:
            raise RuntimeError(f"Stage 1 모델 로드 실패: {e}")

        # Stage 2: 4가지 번아웃 카테고리 분류
        s2_path = f"{Config.MODEL_DIR}/stage2_model.pt"
        try:
            s2_ckpt = torch.load(s2_path, map_location=Config.DEVICE, weights_only=False)
            self.stage2 = BurnoutClassifier(
                input_dim=s2_ckpt.get('embedding_dim', 1024),
                hidden_dim=s2_ckpt.get('hidden_dim', 256),
                num_classes=4
            ).to(Config.DEVICE)
            self.stage2.load_state_dict(s2_ckpt['model_state_dict'])
            self.stage2.eval()
        except FileNotFoundError:
            raise RuntimeError(f"Stage 2 모델 파일 없음: {s2_path}")
        except Exception as e:
            raise RuntimeError(f"Stage 2 모델 로드 실패: {e}")

        self._initialized = True
        print("모델 로딩 완료!")

    # --------------------------------------------------
    # 내부 유틸
    # --------------------------------------------------

    def _get_embedding(self, text: str) -> torch.Tensor:
        """텍스트 단건 임베딩 (analyze 내부용)"""
        return self.kure.encode(text, convert_to_tensor=True).unsqueeze(0).to(Config.DEVICE)

    def _get_embeddings_batch(self, texts: List[str]) -> torch.Tensor:
        """텍스트 리스트 배치 임베딩 — encode() 한 번으로 N개 처리"""
        embs = self.kure.encode(texts, convert_to_tensor=True, batch_size=32)
        return embs.to(Config.DEVICE)  # shape: (N, embedding_dim)

    def _prepare_text(self, text: str, keywords: Optional[Dict]) -> str:
        """텍스트 + 키워드 합산 전처리"""
        analysis_text = text or ""
        if keywords:
            kw_str = " ".join(
                f"{k}: {v}" if isinstance(v, str) else str(v)
                for k, v in keywords.items()
            )
            analysis_text = f"{analysis_text} {kw_str}".strip()
        return analysis_text

    def _empty_result(self) -> Dict:
        s2_probs_map = {cat: -1.0 for cat in STAGE2_CATEGORIES.values()}
        return {
            "primary_emotion": "긍정",
            "primary_score": 0.5,
            "mbi_category": "NORMAL",
            "emotion_probs": {"긍정": 0.5, "부정": 0.5, **s2_probs_map},
            "burnout_category": None,
            "keywords": []
        }

    # --------------------------------------------------
    # 단건 분류 (analyze 내부용)
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 단건 분석 (기존 API, 하위 호환 유지)
    # --------------------------------------------------

    def analyze(self, text: str, keywords: Optional[Dict] = None) -> Dict:
        """전체 분석 파이프라인 (단건)"""
        analysis_text = self._prepare_text(text, keywords)

        if not analysis_text:
            return self._empty_result()

        # Stage 1
        s1_pred, s1_probs = self.predict_stage1(analysis_text)
        primary_emotion = STAGE1_CATEGORIES[s1_pred]
        s2_probs_map = {cat: -1.0 for cat in STAGE2_CATEGORIES.values()}

        result = {
            "primary_emotion": primary_emotion,
            "primary_score": float(s1_probs[s1_pred]),
            "emotion_probs": {"긍정": float(s1_probs[0]), "부정": float(s1_probs[1]), **s2_probs_map},
            "burnout_category": None,
            "mbi_category": "NORMAL",
            "keywords": []
        }

        # Stage 2 (부정인 경우만)
        if s1_pred == 1:
            s2_pred, s2_probs = self.predict_stage2(analysis_text)
            burnout_category = STAGE2_CATEGORIES[s2_pred]
            result["burnout_category"] = burnout_category
            result["mbi_category"] = burnout_category
            result["keywords"] = self.extract_keywords(analysis_text, burnout_category)
            for i, cat in STAGE2_CATEGORIES.items():
                result["emotion_probs"][cat] = round(float(s2_probs[i]), 4)

        return result

    # --------------------------------------------------
    # 배치 분석 (히스토리 N개를 한 번에 처리)
    # --------------------------------------------------

    def analyze_batch(self, items: List[Dict]) -> List[Dict]:
        """
        히스토리 일기 N개를 배치로 분석.

        items: [{"text": str, "keywords": dict | None}, ...]
        반환: analyze()와 동일한 구조의 딕셔너리 리스트

        임베딩을 한 번에 수행 → GPU 전송 오버헤드 N배 → 1번으로 단축.
        Stage 1 MLP, Stage 2 MLP도 배치 forward 처리.
        """
        if not items:
            return []

        # 텍스트 전처리
        texts = [self._prepare_text(item.get("text", ""), item.get("keywords")) for item in items]

        # 빈 텍스트 인덱스 분리
        empty_mask = [t == "" for t in texts]
        non_empty_texts = [t for t, empty in zip(texts, empty_mask) if not empty]
        non_empty_indices = [i for i, empty in enumerate(empty_mask) if not empty]

        # 결과 슬롯 초기화
        results = [self._empty_result() for _ in items]

        if not non_empty_texts:
            return results

        # ── Stage 1 배치 임베딩 + forward ──
        with torch.no_grad():
            embs = self._get_embeddings_batch(non_empty_texts)          # (M, D)
            s1_logits = self.stage1(embs)                               # (M, 2)
            s1_probs_all = F.softmax(s1_logits, dim=-1).cpu().numpy()   # (M, 2)
            s1_preds = np.argmax(s1_probs_all, axis=1)                  # (M,)

        # 부정 샘플만 Stage 2 배치
        neg_local_indices = [i for i, pred in enumerate(s1_preds) if pred == 1]
        neg_embs = embs[neg_local_indices] if neg_local_indices else None

        s2_probs_all = None
        s2_preds = None
        if neg_local_indices and neg_embs is not None:
            with torch.no_grad():
                s2_logits = self.stage2(neg_embs)                           # (K, 4)
                s2_probs_all = F.softmax(s2_logits, dim=-1).cpu().numpy()   # (K, 4)
                s2_preds = np.argmax(s2_probs_all, axis=1)                  # (K,)

        # ── 결과 조립 ──
        neg_cursor = 0
        for local_i, global_i in enumerate(non_empty_indices):
            s1_pred = int(s1_preds[local_i])
            s1_probs = s1_probs_all[local_i]
            primary_emotion = STAGE1_CATEGORIES[s1_pred]
            s2_probs_map = {cat: -1.0 for cat in STAGE2_CATEGORIES.values()}

            r = {
                "primary_emotion": primary_emotion,
                "primary_score": float(s1_probs[s1_pred]),
                "emotion_probs": {"긍정": float(s1_probs[0]), "부정": float(s1_probs[1]), **s2_probs_map},
                "burnout_category": None,
                "mbi_category": "NORMAL",
                "keywords": []
            }

            if s1_pred == 1 and s2_probs_all is not None:
                s2_pred = int(s2_preds[neg_cursor])
                s2_probs = s2_probs_all[neg_cursor]
                neg_cursor += 1
                burnout_category = STAGE2_CATEGORIES[s2_pred]
                r["burnout_category"] = burnout_category
                r["mbi_category"] = burnout_category
                r["keywords"] = self.extract_keywords(texts[global_i], burnout_category)
                for i, cat in STAGE2_CATEGORIES.items():
                    r["emotion_probs"][cat] = round(float(s2_probs[i]), 4)

            results[global_i] = r

        return results
