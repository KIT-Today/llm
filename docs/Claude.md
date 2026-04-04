# Claude.md — AI 인스턴스 간 공유 컨텍스트

> 이 파일은 Claude가 새 대화를 시작할 때 프로젝트 상태를 빠르게 파악하기 위한 파일입니다.
> 코드 변경사항은 `CHANGES.md`, API 명세는 `API_SPEC.md`를 참고하세요.
> **충돌 시 최신 내용으로 대체합니다.**

---

## 프로젝트 개요

- **앱 이름**: 오늘도
- **주제**: 한국형 번아웃 감지 + 마음챙김 앱
- **팀 구성**: 3인 (조민성: AI 서버, 신효주: 프론트엔드, 권윤정: 백엔드)
- **지도교수**: 영상처리 전공 — 딥러닝 analogy(도메인 적응 등) 브리징 설명이 효과적

---

## AI 서버 구조

```
llm/
├── ai_server.py       # FastAPI 메인 (v2.9)
├── analyzer.py        # KURE 임베딩 + 2단계 분류
├── feedback.py        # 템플릿/LLM 피드백 생성
├── emotion_match.py   # 감정 일치도 검사
├── insight.py         # 통계 인사이트
├── error_codes.py     # 에러 코드 (AI1xxx~AI5xxx)
├── config.py          # 환경변수
├── constants.py       # 상수 (페르소나, 활동 45개 등)
├── models.py          # Pydantic 모델
├── scripts/
│   └── generate_diary_data.py   # 합성 일기체 데이터 생성 (완료)
├── models/                      # 학습 완료 모델 저장
├── checkpoints/                 # 에폭 체크포인트 (학습 재개용)
├── dataset/
│   ├── stage2_train_v3.csv      # 원본 구어체 (39,547건)
│   ├── stage2_val_v3.csv        # 검증셋 (4,395건)
│   └── diary_synthetic.csv      # 합성 일기체 (4,026건)
├── docs/
│   ├── CHANGES.md         # 코드 변경 이력 ← 핵심
│   ├── NOTEBOOK_CHANGES.md
│   ├── API_SPEC.md
│   ├── DEPLOYMENT.md
│   └── Claude.md          # 이 파일
└── notebooks/
    ├── KURE_Burnout_2Stage_v3.ipynb
    ├── KURE_Burnout_FineTune_v2.ipynb
    ├── KURE_Burnout_StyleTransfer_v2.ipynb
    ├── KURE_Burnout_E2E_v1.ipynb
    ├── KURE_Burnout_SyntheticData_v1.ipynb  # 합성 혼합 실험 (완료)
    └── KURE_Burnout_SyntheticData_v2.ipynb  # 고비율 실험 (5:1/7:1, 데이터 증량 후 실행 예정)
```

---

## 모델 파이프라인

```
입력 텍스트
  ↓
[Stage 1] KURE 임베딩 → MLP 헤드 → 긍정 / 부정   (F1 ≈ 0.9877)
  ↓ 부정인 경우만
[Stage 2] KURE 임베딩 → MLP 헤드 → 4개 번아웃 카테고리   (F1 ≈ 0.4839)
```

### MBI 카테고리 (4개)

| 코드 | 한국어 |
|------|--------|
| `NORMAL` | 긍정 (번아웃 없음) |
| `정서적_고갈` | 에너지 소진, 피로 |
| `좌절_압박` | 분노, 불만, 억울함 |
| `부정적_대인관계` | 대인 갈등, 소외감 |
| `자기비하` | 불안, 자책, 수치심 |

### 현재 최고 성능 모델

| 파일 | 설명 | Stage 2 F1 |
|------|------|------------|
| `stage1_model_v3.pt` | Stage 1 (frozen) | — |
| `stage2_model_syn_1to7.pt` | Synthetic v3 1:7 (현재 최고) | **0.4849** |
| `stage2_model_v3_ft.pt` | FineTune v4 (기존 운영) | 0.4839 |
| `stage2_model_st_v2.pt` | StyleTransfer v2 | 0.4690 (하락, 미사용) |

---

## 핵심 병목 및 현재 전략

### 문제: Stage 2 F1 정체 (0.47~0.48)
- **근본 원인**: 학습 데이터(AI Hub 구어체) vs 실사용자 입력(일기체) 도메인 미스매치
- 공개된 한국어 번아웃 일기체 데이터셋 없음

### 시도한 접근 (전체 이력)

| 접근 | 결과 | 비고 |
|------|------|------|
| 스타일 트랜스퍼 v1 (EEVE) | A100 OOM | T4 불가 |
| 스타일 트랜스퍼 v2 (Qwen) | F1 0.4690 | 오히려 하락, 폐기 |
| E2E 파인튜닝 v1 (레이어 2) | F1 0.4835 | FineTune v4와 거의 동일 |
| E2E 파인튜닝 v2 (레이어 4) | F1 0.4721 | 레이어 많이 풀수록 하락 |
| 합성 데이터 혼합 v1 (1:3~3:1) | F1 0.4770~0.4835 | 데이터 부족으로 결과 무효 (학습 노이즈) |
| **합성 데이터 혼합 v3 (1:7)** | **F1 0.4849** | **기존 최고 돌파 ★** |

> ⚠️ 스타일 트랜스퍼(기존 데이터 변환)와 합성 데이터 생성(처음부터 새로 생성)은 다른 접근임.
> 둘 다 Qwen을 사용했지만, 스타일 트랜스퍼는 실패·폐기됐고 합성 데이터는 별개 실험.

### 현재 상태 (2026-03-31)
- **합성 데이터 증량 완료**: 총 ~28,026건 (카테고리당 ~7,006건)
- **Synthetic v3 실험 완료** (7개 비율, Colab)
  - 최고: 1:7 (F1=0.4849) — 기존 최고(FineTune v4 0.4839) 돌파
  - 합성 비율 증가 → 성능 단조 감소 확인
  - ⚠️ warm-start random init (Colab 경로 불일치) → 재실험 시 추가 개선 가능성

### 데이터 현황 (실제 수치)
| 파일 | 건수 | 비고 |
|------|------|------|
| stage2_train_v3.csv | 39,547 | 원본 구어체, 클래스 균등 (~25%씩) |
| stage2_val_v3.csv | 4,395 | 검증셋, 원본 고정 |
| diary_synthetic.csv | 4,026→24,000 | 합성 일기체, 증량 중 |

### 원본 데이터셋 수치 (2026-03-29 확인)
| 데이터셋 | 원본 건수 |
|---------|---------|
| 감성대화 (Train+Val) | 58,268건 |
| 웰니스 (두 버전 합산) | 25,000건 |
| 한국어 감정 연속 대화 | 55,629건 |
| **원본 합계** | **약 139,000건** |
| **전처리 후 (Stage2 전체)** | **43,942건** (생존율 ~32%) |

> ⚠️ 이전에 "7.5만 건"으로 추정했으나 오류. 원본은 약 14만 건, 번아웃 카테고리 필터 후 4.4만 건.

### 다음 단계
1. warm-start 경로 수정 후 1:7 재실험 (추가 개선 기대)
2. 1:9, 1:11 등 더 낮은 합성 비율 탐색

### Contribution 프레이밍
- "공개 일기체 데이터 부재 환경에서 LLM 합성 + 일기체 검증 필터 + KURE E2E 파인튜닝을 결합한 도메인 적응 파이프라인 제안"
- 교수님 키워드: **도메인 적응(Domain Adaptation)**

---

## 노트북 스펙: `KURE_Burnout_SyntheticData_v1.ipynb` (구현 완료, 실행 중)

> 기존 노트북(특히 `KURE_Burnout_E2E_v1.ipynb`)의 구조를 재사용.

### 목적
합성 일기체 데이터(`diary_synthetic.csv`)를 원본 데이터(`stage2_train_v3.csv`)와 혼합하여
Stage 2 분류기를 재학습. 혼합 비율 3가지를 순차 실험해 최적값 탐색.

### Drive 경로
```
DATA_PATH      = "/content/drive/MyDrive/Burnout"
DATASET_PATH   = f"{DATA_PATH}/dataset"
PROCESSED_PATH = f"{DATASET_PATH}/processed"

# 입력
ORIGINAL_CSV   = f"{DATA_PATH}/stage2_train_v3.csv"       # 원본 구어체 (39,547건)
SYNTHETIC_CSV  = f"{DATA_PATH}/diary_synthetic.csv"        # 합성 일기체 (4,000건)
VAL_CSV        = f"{DATA_PATH}/stage2_val_v3.csv"          # 검증셋 (원본, 고정)
BASE_MODEL_PATH = f"{DATA_PATH}/stage2_model_v3_ft.pt"     # warm-start 기준

# 출력 (비율별)
SAVE_PATHS = {
    '1:3': f"{DATA_PATH}/stage2_model_synmix_1to3.pt",
    '1:1': f"{DATA_PATH}/stage2_model_synmix_1to1.pt",
    '3:1': f"{DATA_PATH}/stage2_model_synmix_3to1.pt",
}
RESULT_CSV = f"{DATA_PATH}/synmix_results.csv"   # 비율별 F1 비교표
```

### 단계별 구성

| 섹션 | 내용 |
|------|------|
| 1 | 환경 설정 (pip install, imports) |
| 2 | Drive 마운트 + 경로 확인 |
| 3 | 데이터 로드 및 확인 (원본/합성/검증 분포 출력) |
| 4 | MLP 분류기 정의 (FineTune_v2와 동일 아키텍처) |
| 5 | KURE 로드 (frozen, sentence-transformers) |
| 6 | 임베딩 캐시 생성 (원본/합성/검증 각각) |
| 7 | 혼합 비율 실험 루프 (1:3 → 1:1 → 3:1 순서) |
| 8 | 결과 비교표 출력 + CSV 저장 |
| 9 | 직접 문장 테스트 (기존 노트북과 동일한 TEST_SENTENCES 사용) |

### 핵심 구현 사항

**혼합 로직** (섹션 7 내부):
```python
for ratio_name, (syn_ratio, orig_ratio) in RATIOS.items():
    # syn_ratio : orig_ratio 비율로 합성/원본 샘플링
    n_synthetic = int(len(synthetic_df) * syn_ratio / (syn_ratio + orig_ratio) * TOTAL_TRAIN_SIZE / len(synthetic_df) * len(synthetic_df))
    # ... 클래스 균형 유지하면서 샘플링
    # warm-start from stage2_model_v3_ft.pt
    # Focal Loss (gamma=2.0) + class weights
    # epochs=60, patience=10, lr=1e-4, dropout=0.2
    # 저장 후 다음 비율로 이어서 실험
```

**학습 설정** (FineTune_v2와 동일하게 유지):
```python
CONFIG = {
    'epochs': 60, 'batch_size': 64, 'lr': 1e-4, 'weight_decay': 1e-4,
    'patience': 10, 'warmup_epochs': 3, 'focal_gamma': 2.0,
    'label_smoothing': 0.05, 'dropout': 0.2,
    'embedding_dim': 4096,   # KURE 출력 차원
    'hidden_dim': 256,
}
```

**비율 정의**:
```python
RATIOS = {
    '1:3': (1, 3),   # 합성 25% : 원본 75%
    '1:1': (1, 1),   # 합성 50% : 원본 50%
    '3:1': (3, 1),   # 합성 75% : 원본 25%
}
```

**결과 비교표 형식** (섹션 8):
```
혼합 비율    F1 (macro)    v3 대비    FineTune_v4 대비
1:3          0.XXXX        +X.XXXX    +X.XXXX
1:1          0.XXXX        +X.XXXX    +X.XXXX
3:1          0.XXXX        +X.XXXX    +X.XXXX
기준 (v3)    0.4754        —          —
FineTune_v4  0.4839        +0.0085    —
```

### 주의사항
- 검증셋(`stage2_val_v3.csv`)은 **원본 구어체 고정** — 합성 데이터 포함 금지
- 각 비율 실험 후 모델 저장, 다음 비율 시작 전 모델 초기화 (warm-start 재로드)
- `data_version` 필드: `'synthetic_mix_1to3'` / `'synthetic_mix_1to1'` / `'synthetic_mix_3to1'`
- 클래스 불균형 대응: 합성 데이터는 균형(각 1,000건)이지만 원본은 불균형 → class_weights 재계산 필수

---

## 일기체 합성 데이터 생성 — 설계 메모

> `scripts/generate_diary_data.py` 구현 완료. 이 섹션은 설계 근거 참고용.

### 핵심 문제
LLM에게 "일기체로 써줘"라고 해도 실제로 일기체가 나온다는 보장이 없음.
→ **일기체 여부를 검증하는 스코어러가 필수**

### DiaryStyleScorer 설계

**점수 체계**: 각 항목 가중치 합산, 임계값(기본 2.0) 이상만 통과

```
[+2] 과거형 종결어미: ~했다, ~었다, ~였다, ~겠다
[+1] 일기체 시작 패턴: "오늘", "나는", "내가", "그냥", "진짜", "너무"
[+1] 독백성 표현: "~것 같다", "~싶다", "~모르겠다"
[+1] 자기 서술 주어: "나", "내"
[-2] 구어체 종결어미: ~요, ~죠, ~어요, ~세요
[-1] 대화 반응어: "아", "어", "헐", "맞아", "그래" (문장 시작)
[-1] 상대방 지칭: "당신", "너는", "저는" (대화체 신호)
```

### 전체 필터 파이프라인

```
Ollama 생성
  → 기본 후처리 (길이, 불완전 문장, 프롬프트 누출)
  → DiaryStyleScorer (일기체 점수 임계값 통과 여부)
  → 중복 제거 (seen_texts set, 기존 CSV 포함)
  → 카테고리 일관성 검사 (--use-consistency 활성화 시만, 기본 비활성)
  → 통과 시 CSV 저장
```

---

## 아이디어 메모 (우선순위 낮음, 검토 보류)

### Discriminator 학습 가능화 (부트스트래핑)
- 현재 `DiaryStyleScorer`는 룰 기반 고정. 이걸 학습 가능한 분류기로 교체하는 아이디어.
- 구조: Generator(Ollama LLM, 고정) + Discriminator(학습 가능 분류기) → KoELECTRA / GAN에서 영감받은 구조
- 흐름: 룰 기반으로 초기 데이터 생성 → Discriminator 학습 → 더 정교한 필터로 재생성 → 반복 (부트스트래핑)
- 한계: Generator 고정이라 진짜 GAN 아님. Discriminator 학습용 레이블링 공수 필요. 순환논리 리스크.
- 활용: Future work 또는 교수님 상담 시 "발전 방향"으로 언급하는 정도

### 검증 데이터 독립성 확보
- 학습용(Qwen)과 검증용 데이터를 다른 모델(ex. llama3)로 생성 → Generator 다변화로 평가 독립성 확보
- `generate_diary_data.py --model llama3:8b --output diary_holdout.csv` 형태로 현재 스크립트 재활용 가능
- 효과: "같은 파이프라인으로 자기 자신을 평가한 게 아니다" → accuracy 설득력 상승
- 시점: 합성 데이터 혼합 재학습 완료 후 진행해도 늦지 않음

---

## 미결정 사항

| 항목 | 내용 |
|------|------|
| MBI 구조 통일 | 제안서의 MBI 3단계(EE/DP/PA) vs 코드의 4카테고리 독립 분류 |
| 일기 vs 챗봇 | 교수님이 챗봇 전환 직접 제안 (2026-03-11 발표 피드백) — 팀 논의 필요 |
| KoAlpaca 제안서 반영 | 제안서에 기재됐으나 미구현 → KoAlpaca 합성 방향으로 구현 예정 |
| 개인정보 동의 플로우 | 온보딩 동의 절차 미설계 |
| 혼합 비율 최적값 | 현재 1:7 최고(F1=0.4849). warm-start 적용 및 1:9/1:11 탐색 후 확정 |

---

## 주요 상수 / 설정

- **페르소나**: 정수 1~5로 저장 (1=따뜻한 상담사, 2=실용적 조언자, 3=친근한 친구, 4=차분한 멘토, 5=밝은 응원단)
- **활동 DB**: 45개 (REST 1~15 / VENTILATION 16~30 / SMALL_WIN 31~45)
- **최소 일기 수**: 3개 미만 시 `mbi_category = NONE`, recommendations 미포함
- **피드백 배치**: 2주마다 백엔드 → AI 서버, `feedback_data.csv` 누적
- **USE_LLM**: 기본 `false` (템플릿), `true` 시 KoAlpaca 활성화

---

## API 안정성

v2.7~v2.9 모든 변경은 **내부 구현 개선**이며 외부 API 계약은 변경 없음.
백엔드/프론트엔드 연동 코드 수정 불필요.

| 버전 | 변경 성격 | API 영향 |
|------|-----------|----------|
| v2.7 | 취소 엔드포인트 추가 | 신규 엔드포인트 추가 (기존 호환) |
| v2.8 | Graceful 모델 로드 | 없음 |
| v2.9 | 분석 큐 구조 교체 | 없음 |

---

## 주의사항

- `stage1_model.pt` / `stage2_model.pt` 는 llm/ 루트에 위치해야 서버 로드 가능
- KURE 백본은 항상 frozen (E2E 실험 제외)
- `mbi_category` 값은 **한국어**로 통일 (`정서적_고갈` 등), `NORMAL`만 영문 예외
- Stage 2 미실행 시 카테고리 확률은 `-1.0` 센티널 (0.0과 구분)
- 콜백 실패 시 `error_codes.py`의 AI4xxx 계열 에러 확인

---

## ~~미구현~~ 완료 — 분석 큐 구조 (v2.9, 2026-04-04)

> **배경**: `BackgroundTasks`는 동시성 제어 없음 → 생성-삭제를 빠르게 반복하면 분석 태스크가 무한 쌓여 GPU OOM / 서버 freeze 가능.

### 설계 방향: deque 기반 커스텀 큐 + 단일 worker

- `/analyze` → 큐에 태스크 추가
- `/analysis/cancel/{diary_id}` → 큐에서 해당 diary_id 항목 즉시 제거 (추론 자체를 안 함)
- worker 1개가 큐에서 순서대로 꺼내 처리 → GPU 동시 접근 없음
- `lifespan`에서 worker를 `asyncio.create_task()`로 백그라운드 상시 실행

```python
from collections import deque
import asyncio

analysis_queue: deque = deque()          # (diary_id, user_id, persona, history) 튜플
analysis_queue_lock = asyncio.Lock()     # deque 접근 동기화
analysis_worker_task = None              # lifespan에서 관리

async def analysis_worker():
    while True:
        async with analysis_queue_lock:
            if analysis_queue:
                item = analysis_queue.popleft()
            else:
                item = None
        if item:
            await process_analysis(*item)
        else:
            await asyncio.sleep(0.1)

# /analyze
async with analysis_queue_lock:
    analysis_queue.append((diary_id, user_id, persona, history))

# /analysis/cancel/{diary_id}
async with analysis_queue_lock:
    # deque를 순회하며 해당 diary_id 제거
    to_keep = [item for item in analysis_queue if item[0] != diary_id]
    analysis_queue.clear()
    analysis_queue.extend(to_keep)
```

**주의**: 현재 `cancelled_diary_ids` 플래그도 함께 유지 — worker가 꺼낸 직후 ~ 처리 시작 전 타이밍 커버용.

---

## ~~미구현~~ 완료 — Graceful 처리 (v2.8, 2026-04-04)

> 현재 `analyzer.py`의 `initialize()`에 예외 처리 없음 → 모델 파일 없으면 서버 크래시.
> 아래 로직을 구현해야 배포 환경에서 안전하게 동작함.

### 1. `analyzer.py` — `initialize()` 예외 분리

각 단계(KURE / Stage1 / Stage2)를 개별 try/except로 감싸고 명확한 `RuntimeError` 메시지 raise:

```python
def initialize(self):
    if self._initialized:
        return
    try:
        self.kure = SentenceTransformer("nlpai-lab/KURE-v1", device=Config.DEVICE)
    except Exception as e:
        raise RuntimeError(f"KURE 임베딩 모델 로드 실패: {e}")
    try:
        s1_path = f"{Config.MODEL_DIR}/stage1_model.pt"
        s1_ckpt = torch.load(s1_path, map_location=Config.DEVICE, weights_only=False)
        self.stage1 = BurnoutClassifier(...).to(Config.DEVICE)
        self.stage1.load_state_dict(s1_ckpt['model_state_dict'])
        self.stage1.eval()
    except FileNotFoundError:
        raise RuntimeError(f"Stage 1 모델 파일 없음: {s1_path}")
    except Exception as e:
        raise RuntimeError(f"Stage 1 모델 로드 실패: {e}")
    # Stage 2도 동일 패턴
    self._initialized = True
```

### 2. `ai_server.py` — `lifespan()` graceful 처리

모델 로드 실패해도 서버는 뜨게 하고, 분석 요청은 기존 `MODEL_NOT_LOADED` 에러코드 경로 타도록:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer, ...
    try:
        analyzer = BurnoutAnalyzer()
        analyzer.initialize()
    except RuntimeError as e:
        print(f"[경고] 모델 로드 실패: {e}")
        print("[경고] 서버는 실행되지만 분석 요청은 실패합니다.")
        analyzer = BurnoutAnalyzer()  # _initialized=False 상태 유지
    # 나머지 초기화는 그대로
    ...
    yield
```

**효과**: 모델 파일 없어도 서버 기동 → `/analyze` 요청 시 `process_analysis`의 `MODEL_NOT_LOADED` 체크 → fallback 콜백 전송까지 정상 동작

---

## ~~미구현~~ 완료 — Graceful 처리 (v2.8, 2026-04-04)

> 현재 `analyzer.py`의 `initialize()`에 예외 처리 없음 → 모델 파일 없으면 서버 크래시.
> 아래 로직을 구현해야 배포 환경에서 안전하게 동작함.

### 1. `analyzer.py` — `initialize()` 예외 분리

각 단계(KURE / Stage1 / Stage2)를 개별 try/except로 감싸고 명확한 `RuntimeError` 메시지 raise:

```python
def initialize(self):
    if self._initialized:
        return
    try:
        self.kure = SentenceTransformer("nlpai-lab/KURE-v1", device=Config.DEVICE)
    except Exception as e:
        raise RuntimeError(f"KURE 임베딩 모델 로드 실패: {e}")
    try:
        s1_path = f"{Config.MODEL_DIR}/stage1_model.pt"
        s1_ckpt = torch.load(s1_path, map_location=Config.DEVICE, weights_only=False)
        self.stage1 = BurnoutClassifier(...).to(Config.DEVICE)
        self.stage1.load_state_dict(s1_ckpt['model_state_dict'])
        self.stage1.eval()
    except FileNotFoundError:
        raise RuntimeError(f"Stage 1 모델 파일 없음: {s1_path}")
    except Exception as e:
        raise RuntimeError(f"Stage 1 모델 로드 실패: {e}")
    # Stage 2도 동일 패턴
    self._initialized = True
```

### 2. `ai_server.py` — `lifespan()` graceful 처리

모델 로드 실패해도 서버는 뜨게 하고, 분석 요청은 기존 `MODEL_NOT_LOADED` 에러코드 경로 타도록:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer, ...
    try:
        analyzer = BurnoutAnalyzer()
        analyzer.initialize()
    except RuntimeError as e:
        print(f"[경고] 모델 로드 실패: {e}")
        print("[경고] 서버는 실행되지만 분석 요청은 실패합니다.")
        analyzer = BurnoutAnalyzer()  # _initialized=False 상태 유지
    # 나머지 초기화는 그대로
    ...
    yield
```

**효과**: 모델 파일 없어도 서버 기동 → `/analyze` 요청 시 `process_analysis`의 `MODEL_NOT_LOADED` 체크 → fallback 콜백 전송까지 정상 동작
