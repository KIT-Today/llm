# -*- coding: utf-8 -*-
"""
번아웃 일기체 합성 데이터 생성 스크립트

사용법:
    python scripts/generate_diary_data.py
    python scripts/generate_diary_data.py --target 500 --model qwen2.5:7b
    python scripts/generate_diary_data.py --category 정서적_고갈 --target 200
    python scripts/generate_diary_data.py --use-consistency   # Stage 2 모델로 카테고리 검사 활성화

출력:
    diary_synthetic.csv          — 생성된 일기체 문장
    diary_gen_checkpoint.json    — 중단/재시작 지원 체크포인트

필터 파이프라인:
    Ollama 생성
      → 기본 후처리 (길이, 완전성, 프롬프트 누출, 외국어)
      → DiaryStyleScorer (일기체 점수 임계값)
      → CategoryConsistencyChecker (Stage 2 모델 보조 필터, 선택적)
      → CSV 저장
"""

import sys
import os

# llm/ 루트를 path에 추가 (analyzer 임포트용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import csv
import random
import argparse
import requests
import time
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from analyzer import BurnoutClassifier


# ============================================================
# 설정 상수
# ============================================================

OLLAMA_URL        = "http://localhost:11434/api/generate"
DEFAULT_MODEL     = "qwen2.5:7b"
DEFAULT_TARGET     = 6000
DEFAULT_THRESHOLD  = 2.0
DEFAULT_BATCH_SIZE = 10
CHECKPOINT_EVERY   = 50

BASE_DIR = Path(__file__).resolve().parent.parent  # llm/

CATEGORIES: Dict[str, int] = {
    "정서적_고갈":     0,
    "좌절_압박":       1,
    "부정적_대인관계": 2,
    "자기비하":        3,
}


# ============================================================
# Ollama 프롬프트
# ============================================================

SYSTEM_PROMPT = (
    "당신은 한국어 일기 작성 전문가입니다.\n"
    "주어진 감정 상황에 맞는 짧은 일기 문장을 작성하세요.\n\n"
    "규칙:\n"
    "- 반드시 일기체: ~했다, ~었다, ~다 등의 종결어미 사용\n"
    "- 1~3문장, 30~120글자\n"
    "- 오늘 하루의 감정을 독백하는 형식 (1인칭 나, 내)\n"
    "- 존댓말(~요, ~어요, ~세요) 절대 사용 금지\n"
    "- 설명, 인사말 없이 일기 본문만 출력\n"
    "- 중국어, 영어 등 다른 언어 사용 금지"
)

CATEGORY_USER_PROMPTS: Dict[str, str] = {
    "정서적_고갈": (
        "오늘 하루 몸도 마음도 완전히 지치고 탈진한 사람의 일기를 써줘. "
        "피로, 무기력, 에너지 고갈 감정을 표현해줘."
    ),
    "좌절_압박": (
        "억울하고 화나고 답답한 감정을 담은 일기를 써줘. "
        "분노, 불만, 압박감, 스트레스를 표현해줘."
    ),
    "부정적_대인관계": (
        "주변 사람들과의 갈등이나 소외감, 외로움을 담은 일기를 써줘. "
        "대인관계 스트레스, 배신감, 외로움을 표현해줘."
    ),
    "자기비하": (
        "자신에 대한 불안, 자책, 부족함을 느끼는 일기를 써줘. "
        "자기비하, 후회, 수치심, 무능함을 표현해줘."
    ),
}


def make_batch_prompt(category: str, batch_size: int) -> str:
    """배치 생성용 프롬프트: N개 번호 목록으로 요청"""
    return (
        f"{CATEGORY_USER_PROMPTS[category]}\n\n"
        f"서로 다른 상황과 표현으로 {batch_size}개 작성해줘.\n"
        f"형식: 번호. 일기 내용 (각 줄에 하나씩, 번호 외 설명 없이)"
    )


# ============================================================
# DiaryStyleScorer (Claude.md 설계 기반)
# ============================================================

class DiaryStyleScorer:
    """
    일기체 여부 점수화 스코어러

    점수 체계:
        [+2] 과거형 종결어미: ~했다, ~었다, ~였다, ~겠다
        [+1] 일기체 시작 패턴: 오늘, 나는, 내가, 그냥, 진짜, 너무
        [+1] 독백성 표현: ~것 같다, ~싶다, ~모르겠다
        [+1] 자기 서술 주어: 나, 내 (최대 2점)
        [-2] 구어체 종결어미: ~요, ~죠, ~어요, ~세요
        [-1] 대화 반응어 (문장 시작): 아, 어, 헐, 맞아, 그래
        [-1] 상대방 지칭: 당신, 너는, 저는

    점수는 문장 길이(40자 기준)로 정규화 — 긴 문장 불이익 방지
    """

    _PAST       = re.compile(r'(했다|었다|였다|겠다)(?=[.!?,\s]|$)')
    _STARTER    = re.compile(r'^(오늘|나는|내가|그냥|진짜|너무)')
    _MONOLOGUE  = re.compile(r'것 같다|싶다|모르겠다')
    _SELF       = re.compile(r'나[는가]|내[가]?')
    _SPOKEN     = re.compile(r'(세요|어요|죠|요)(?=[.!?,\s]|$)')
    _REACTION   = re.compile(r'(?:^|(?<=[.!?\n])\s*)(아|어|헐|맞아|그래)[,. ]')
    _SECOND_P   = re.compile(r'당신|너는|저는')

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def score(self, text: str) -> float:
        raw  = 0.0
        raw += 2.0 * len(self._PAST.findall(text))
        raw += 1.0 * (1 if self._STARTER.search(text) else 0)
        raw += 1.0 * len(self._MONOLOGUE.findall(text))
        raw += 1.0 * min(len(self._SELF.findall(text)), 2)
        raw -= 2.0 * len(self._SPOKEN.findall(text))
        raw -= 1.0 * (1 if self._REACTION.search(text) else 0)
        raw -= 1.0 * len(self._SECOND_P.findall(text))

        # 문장 길이 정규화 (60자 기준 — 80~120자 문장 불이익 완화)
        divisor = max(len(text) / 60.0, 1.0)
        return raw / divisor

    def passes(self, text: str) -> bool:
        return self.score(text) >= self.threshold


# ============================================================
# 기본 후처리
# ============================================================

_PROMPT_LEAK = re.compile(
    r'###\s*응답|규칙:|시스템:|system:|assistant:|당신은\s*(한국어|일기)|반드시\s*일기체',
    re.IGNORECASE,
)
_NON_KOREAN = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff]')  # 중국어·일본어


def postprocess(raw: Optional[str]) -> Optional[str]:
    """생성 텍스트 기본 정제"""
    if not raw:
        return None

    text = raw.strip()

    if _PROMPT_LEAK.search(text):
        return None
    if _NON_KOREAN.search(text):
        return None

    # 이중 개행 또는 줄바꿈 이후 잘라내기 (첫 문단만 사용)
    text = text.split('\n\n')[0].split('\n')[0].strip()

    if len(text) < 10 or len(text) > 200:
        return None

    # 불완전 문장 체크 (마침표 포함)
    if not re.search(r'[요다죠어!?.]$', text):
        return None

    return text


# ============================================================
# Ollama 클라이언트
# ============================================================

_NUMBERED = re.compile(r'^\s*\d+[.)]\s*')


def parse_batch_response(response: str, batch_size: int) -> List[str]:
    """번호 목록 응답 → 개별 문장 리스트"""
    results = []
    for line in response.strip().split('\n'):
        line = _NUMBERED.sub('', line).strip()
        line = re.sub(r'^[-•]\s*', '', line).strip()
        if len(line) >= 10:
            results.append(line)
    return results[:batch_size]


def call_ollama_batch(category: str, model: str, temperature: float,
                      batch_size: int) -> List[str]:
    prompt = f"{SYSTEM_PROMPT}\n\n{make_batch_prompt(category, batch_size)}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.92,
            "num_predict": batch_size * 100,
            "num_gpu": 99,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return parse_batch_response(raw, batch_size)
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Ollama 서버에 연결할 수 없습니다. `ollama serve` 실행 여부 확인")
        raise
    except Exception as e:
        print(f"\n[WARN] Ollama 호출 실패: {e}")
        return []


# ============================================================
# 카테고리 일관성 검사 (보조 필터, 선택적)
# ============================================================

class CategoryConsistencyChecker:
    """
    Stage 2 모델로 카테고리 일관성 검사.
    모델 자체 정확도가 ~47% 이므로 보조 필터 역할만 함.
    모델 파일이 없으면 자동 비활성화.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._active = False
        self._kure   = None
        self._clf    = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def try_load(self) -> bool:
        if not self.model_path.exists():
            print(f"[INFO] Stage 2 모델 없음 ({self.model_path.name}) → 일관성 검사 비활성화")
            return False

        print(f"카테고리 일관성 검사 모델 로딩: {self.model_path.name}")
        ckpt = torch.load(self.model_path, map_location=self._device, weights_only=False)
        self._clf = BurnoutClassifier(
            input_dim=ckpt.get("embedding_dim", 1024),
            hidden_dim=ckpt.get("hidden_dim", 256),
            num_classes=4,
        ).to(self._device)
        self._clf.load_state_dict(ckpt["model_state_dict"])
        self._clf.eval()
        self._kure = SentenceTransformer("nlpai-lab/KURE-v1", device=self._device)
        self._active = True
        print("Stage 2 모델 로딩 완료")
        return True

    def is_consistent(self, text: str, expected_idx: int) -> bool:
        if not self._active:
            return True  # 비활성화 시 항상 통과
        with torch.no_grad():
            emb  = self._kure.encode(text, convert_to_tensor=True).unsqueeze(0).to(self._device)
            pred = int(torch.argmax(F.softmax(self._clf(emb), dim=-1), dim=-1).item())
        return pred == expected_idx


# ============================================================
# 체크포인트
# ============================================================

def load_checkpoint(path: Path) -> Dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for cat in CATEGORIES:
            data.setdefault(cat, [])
        return data
    return {cat: [] for cat in CATEGORIES}


def save_checkpoint(data: Dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 메인 생성 루프
# ============================================================

def load_existing_texts(out_path: Path) -> set:
    """기존 CSV에서 텍스트 목록 로드 (중복 방지용)"""
    seen = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "text" in row:
                    seen.add(row["text"].strip())
    return seen


def generate(args):
    target_cats = [args.category] if args.category else list(CATEGORIES.keys())

    scorer  = DiaryStyleScorer(threshold=args.threshold)
    checker = CategoryConsistencyChecker(BASE_DIR / args.stage2_model)
    if args.use_consistency:
        checker.try_load()

    ckpt_path = Path(args.checkpoint)
    out_path  = Path(args.output)

    collected    = load_checkpoint(ckpt_path)
    seen_texts   = load_existing_texts(out_path)
    write_header = not out_path.exists()

    csv_f  = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(["text", "category", "diary_score"])

    total_gen = total_pass = 0
    run_start = time.time()

    print(f"\n모델: {args.model}  |  카테고리당 목표: {args.target}건  |  임계값: {args.threshold}")
    print(f"출력: {out_path}  |  체크포인트: {ckpt_path}\n")

    try:
        for category in target_cats:
            cat_idx = CATEGORIES[category]
            already = len(collected[category])
            need    = args.target - already

            if need <= 0:
                print(f"[{category}] 완료 ({already}/{args.target}), 건너뜀")
                continue

            print(f"\n{'─'*50}")
            print(f"[{category}]  목표 {args.target} | 기존 {already} | 추가 필요 {need}")
            print(f"{'─'*50}")

            passed = gen = batch_calls = 0
            filter_stats = {"postprocess": 0, "diary_score": 0, "duplicate": 0, "consistency": 0}
            cat_start = time.time()

            while passed < need:
                batch_calls += 1
                temp = random.uniform(0.8, 1.1)

                try:
                    raws = call_ollama_batch(category, args.model, temp, args.batch_size)
                except requests.exceptions.ConnectionError:
                    raise SystemExit(1)

                for raw in raws:
                    if passed >= need:
                        break

                    gen       += 1
                    total_gen += 1

                    # 1. 기본 후처리
                    text = postprocess(raw)
                    if text is None:
                        filter_stats["postprocess"] += 1
                        continue

                    # 2. 일기체 스코어 필터
                    diary_score = scorer.score(text)
                    if not scorer.passes(text):
                        filter_stats["diary_score"] += 1
                        continue

                    # 3. 중복 제거
                    if text in seen_texts:
                        filter_stats["duplicate"] += 1
                        continue

                    # 4. 카테고리 일관성 (보조 필터, --use-consistency 활성화 시)
                    if not checker.is_consistent(text, cat_idx):
                        filter_stats["consistency"] += 1
                        continue

                    # ✓ 통과
                    seen_texts.add(text)
                    collected[category].append(text)
                    writer.writerow([text, category, round(diary_score, 3)])
                    csv_f.flush()
                    passed    += 1
                    total_pass += 1

                total_done = already + passed
                pct        = total_pass / max(total_gen, 1) * 100
                print(
                    f"\r  {total_done}/{args.target}  "
                    f"(호출 {batch_calls}회 | 문장 {gen}개 | 통과율 {pct:.1f}%)",
                    end="", flush=True,
                )

                if batch_calls % 20 == 0:
                    elapsed = time.time() - cat_start
                    rate    = passed / elapsed * 3600 if elapsed > 0 else 0
                    print(
                        f"\n  [로그] 호출 {batch_calls}회 | 문장 {gen}개 | 통과 {passed}건 | "
                        f"탈락 — 후처리:{filter_stats['postprocess']} "
                        f"일기체:{filter_stats['diary_score']} "
                        f"중복:{filter_stats['duplicate']} "
                        f"일관성:{filter_stats['consistency']} | "
                        f"경과 {elapsed/60:.1f}분 | 속도 {rate:.0f}건/시"
                    )

                if total_pass % CHECKPOINT_EVERY == 0 and total_pass > 0:
                    save_checkpoint(collected, ckpt_path)
                    print(f"\n  → 체크포인트 저장 ({total_pass}건)")

            cat_elapsed = time.time() - cat_start
            cat_rate    = need / cat_elapsed * 3600 if cat_elapsed > 0 else 0
            print(f"\n  [{category}] +{need}건 추가 완료 | "
                  f"소요 {cat_elapsed/60:.1f}분 | 평균 {cat_rate:.0f}건/시")
            print(f"  필터 탈락 — 후처리: {filter_stats['postprocess']} | "
                  f"일기체점수: {filter_stats['diary_score']} | "
                  f"중복: {filter_stats['duplicate']} | "
                  f"일관성: {filter_stats['consistency']}")
            save_checkpoint(collected, ckpt_path)

    except KeyboardInterrupt:
        print("\n\n[중단] 체크포인트 저장 중...")
        save_checkpoint(collected, ckpt_path)
    finally:
        csv_f.close()

    total_elapsed = time.time() - run_start
    total_rate    = total_pass / total_elapsed * 3600 if total_elapsed > 0 else 0
    print(f"\n{'='*50}")
    print(f"총 소요 시간: {total_elapsed/60:.1f}분")
    print(f"총 생성 시도: {total_gen}건")
    print(f"필터 통과:   {total_pass}건")
    if total_gen:
        print(f"통과율:      {total_pass / total_gen * 100:.1f}%")
    print(f"평균 속도:   {total_rate:.0f}건/시")
    print(f"저장 위치:   {out_path.resolve()}")
    print(f"{'='*50}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="번아웃 일기체 합성 데이터 생성 (Ollama 기반)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama 모델명 (기본: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--target", type=int, default=DEFAULT_TARGET,
        help=f"카테고리당 목표 건수 (기본: {DEFAULT_TARGET})",
    )
    p.add_argument(
        "--category", choices=list(CATEGORIES.keys()), default=None,
        help="특정 카테고리만 생성 (기본: 전체 4개)",
    )
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"DiaryStyleScorer 임계값 (기본: {DEFAULT_THRESHOLD}, 낮추면 통과율 ↑)",
    )
    p.add_argument(
        "--output", default=str(BASE_DIR / "dataset" / "diary_synthetic.csv"),
        help="출력 CSV 경로 (기본: dataset/diary_synthetic.csv)",
    )
    p.add_argument(
        "--checkpoint", default=str(BASE_DIR / "dataset" / "diary_gen_checkpoint.json"),
        help="체크포인트 파일 경로 (기본: dataset/diary_gen_checkpoint.json)",
    )
    p.add_argument(
        "--stage2-model", default="models/stage2_model_v3.pt",
        help="Stage 2 모델 파일명 (llm/ 기준, 기본: models/stage2_model_v3.pt)",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"호출당 생성 문장 수 (기본: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--use-consistency", action="store_true",
        help="Stage 2 모델로 카테고리 일관성 검사 활성화 (기본: 비활성)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
