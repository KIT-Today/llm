# -*- coding: utf-8 -*-
"""
설정 모듈
"""

import os
import torch


class Config:
    """AI 서버 설정"""
    BACKEND_CALLBACK_URL = os.getenv(
        "BACKEND_CALLBACK_URL", 
        "http://127.0.0.1:8000/diaries/analysis-callback"
    )
    MODEL_DIR = os.getenv("MODEL_DIR", ".")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIN_DIARY_COUNT_FOR_RECOMMENDATION = 3
    MIN_DIARY_COUNT_FOR_INSIGHT = 3
    EMOTION_MISMATCH_THRESHOLD = 0.6
