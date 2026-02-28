# -*- coding: utf-8 -*-
"""
사용자 설문 피드백 저장 모듈

백엔드가 2주마다 배치로 전송하는 피드백을 CSV에 저장하고
카테고리별 오답 분포를 추적합니다.

흐름:
  프론트 → 백엔드(DB 저장) → [2주마다] → AI 서버 POST /feedback/batch
"""

import csv
import os
import threading
from datetime import datetime
from typing import Optional

from config import Config


# ============================================
# 상수
# ============================================

VALID_MBI_CATEGORIES = {"정서적_고갈", "좌절_압박", "부정적_대인관계", "자기비하", "NORMAL"}

CSV_HEADER = [
    "received_at",            # AI 서버가 배치를 수신한 시각
    "predicted_mbi_category", # AI가 예측한 카테고리
    "ai_message_rating",      # AI 메시지 만족도 1~5
    "mbi_category_rating",    # MBI 카테고리 만족도 1~5
]


# ============================================
# FeedbackStore
# ============================================

class FeedbackStore:
    """
    배치 피드백을 CSV에 스레드 안전하게 저장합니다.

    저장 경로: Config.FEEDBACK_CSV_PATH  (기본: feedback_data.csv)
    """

    def __init__(self):
        self._path = Config.FEEDBACK_CSV_PATH
        self._lock = threading.Lock()
        self._ensure_csv()

    # ── 내부 유틸 ──────────────────────────────────────────

    def _ensure_csv(self):
        """CSV 파일이 없으면 헤더 포함 생성"""
        if not os.path.exists(self._path):
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADER)
            print(f"[FeedbackStore] 피드백 CSV 생성: {self._path}")

    def _read_all(self) -> list[dict]:
        """전체 행 읽기 (통계용)"""
        with open(self._path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # ── 퍼블릭 API ─────────────────────────────────────────

    def save_batch(self, records: list[dict]) -> dict:
        """
        배치 피드백을 CSV에 저장합니다.

        Args:
            records: FeedbackRecord 리스트 (dict 형태)

        Returns:
            dict: {received, total_accumulated, avg_ai_message_rating,
                   avg_mbi_category_rating, low_mbi_by_category}
        """
        received_at = datetime.now().isoformat()

        rows = [
            [
                received_at,
                r["predicted_mbi_category"],
                r["ai_message_rating"],
                r["mbi_category_rating"],
            ]
            for r in records
        ]

        with self._lock:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)

        # 이번 배치 통계
        avg_ai  = round(sum(r["ai_message_rating"]    for r in records) / len(records), 2)
        avg_mbi = round(sum(r["mbi_category_rating"]  for r in records) / len(records), 2)
        low_mbi: dict[str, int] = {}
        for r in records:
            if r["mbi_category_rating"] <= 2:
                cat = r["predicted_mbi_category"]
                low_mbi[cat] = low_mbi.get(cat, 0) + 1

        total = self.get_stats()["total"]
        print(
            f"[FeedbackStore] 배치 수신: {len(records)}건 저장 "
            f"(누적: {total}건, AI평점: {avg_ai}, 카테고리평점: {avg_mbi})"
        )

        return {
            "received": len(records),
            "total_accumulated": total,
            "avg_ai_message_rating": avg_ai,
            "avg_mbi_category_rating": avg_mbi,
            "low_mbi_by_category": low_mbi,
        }

    def get_stats(self) -> dict:
        """
        전체 누적 피드백 통계를 반환합니다.

        Returns:
            dict: total, avg_ai_message_rating, avg_mbi_category_rating,
                  low_mbi_by_category
        """
        with self._lock:
            rows = self._read_all()

        total = len(rows)
        if total == 0:
            return {
                "total": 0,
                "avg_ai_message_rating": 0.0,
                "avg_mbi_category_rating": 0.0,
                "low_mbi_by_category": {},
            }

        ai_ratings  = [int(r["ai_message_rating"])   for r in rows if r["ai_message_rating"].isdigit()]
        mbi_ratings = [int(r["mbi_category_rating"]) for r in rows if r["mbi_category_rating"].isdigit()]

        # 카테고리별 낮은 평점(1~2) 수 → 어느 카테고리가 약한지 파악
        low_mbi_by_category: dict[str, int] = {}
        for r in rows:
            if r["mbi_category_rating"].isdigit() and int(r["mbi_category_rating"]) <= 2:
                cat = r["predicted_mbi_category"]
                low_mbi_by_category[cat] = low_mbi_by_category.get(cat, 0) + 1

        return {
            "total": total,
            "avg_ai_message_rating": round(sum(ai_ratings)  / len(ai_ratings),  2) if ai_ratings  else 0.0,
            "avg_mbi_category_rating": round(sum(mbi_ratings) / len(mbi_ratings), 2) if mbi_ratings else 0.0,
            "low_mbi_by_category": low_mbi_by_category,
        }
