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

VALID_MBI_CATEGORIES = {"정서적_고갈", "좌절_압박", "부정적_대인관계", "자기비하", "NONE"}

CSV_HEADER = [
    "received_at",            # AI 서버가 배치를 수신한 시각
    "period_start",           # 해당 배치 기간 시작
    "period_end",             # 해당 배치 기간 끝
    "predicted_mbi_category", # AI가 예측한 카테고리
    "is_correct",             # 사용자 정오 확인
    "satisfaction_score",     # 만족도 1~5
    "user_mbi_category",      # 틀렸을 때 사용자가 선택한 카테고리 (nullable)
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

    def save_batch(
        self,
        period_start: str,
        period_end: str,
        records: list[dict],
    ) -> dict:
        """
        2주 배치 피드백을 CSV에 저장합니다.

        Args:
            period_start: 배치 기간 시작 ("2026-02-01")
            period_end:   배치 기간 끝   ("2026-02-14")
            records:      FeedbackRecord 리스트 (dict 형태)

        Returns:
            dict: {received, total_accumulated, model_accuracy, category_corrections}
        """
        received_at = datetime.now().isoformat()

        rows = [
            [
                received_at,
                period_start,
                period_end,
                r["predicted_mbi_category"],
                r["is_correct"],
                r["satisfaction_score"],
                r.get("user_mbi_category") or "",
            ]
            for r in records
        ]

        with self._lock:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        stats = self.get_stats()
        print(
            f"[FeedbackStore] 배치 수신: {len(records)}건 저장 "
            f"(기간: {period_start} ~ {period_end}, "
            f"누적: {stats['total']}건, 정확도: {stats['model_accuracy']:.1%})"
        )

        return {
            "received": len(records),
            "total_accumulated": stats["total"],
            "model_accuracy": stats["model_accuracy"],
            "category_corrections": stats["category_corrections"],
        }

    def get_stats(self) -> dict:
        """
        전체 누적 피드백 통계를 반환합니다.

        Returns:
            dict: total, correct, incorrect, model_accuracy,
                  avg_satisfaction, category_corrections
        """
        with self._lock:
            rows = self._read_all()

        total = len(rows)
        if total == 0:
            return {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "model_accuracy": 0.0,
                "avg_satisfaction": 0.0,
                "category_corrections": {},
            }

        correct = sum(1 for r in rows if r["is_correct"].lower() == "true")

        satisfaction_scores = [
            int(r["satisfaction_score"])
            for r in rows
            if r["satisfaction_score"].isdigit()
        ]

        # 카테고리별 오답 현황 (어떤 카테고리가 자주 틀리는지)
        category_corrections: dict[str, int] = {}
        for r in rows:
            if r["is_correct"].lower() == "false" and r["user_mbi_category"]:
                cat = r["user_mbi_category"]
                category_corrections[cat] = category_corrections.get(cat, 0) + 1

        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "model_accuracy": round(correct / total, 4),
            "avg_satisfaction": round(
                sum(satisfaction_scores) / len(satisfaction_scores), 2
            ) if satisfaction_scores else 0.0,
            "category_corrections": category_corrections,
        }
