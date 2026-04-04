# 변경사항

## v2.9 — 2026-04-04 분석 큐 구조 (deque + 단일 worker)

> **API 변경 없음** — 엔드포인트 경로·요청·응답 스펙 동일. 내부 처리 방식만 변경.

### 배경
`BackgroundTasks`는 동시성 제어 없음 → 생성-삭제를 빠르게 반복하면 분석 태스크가 무한 쌓여 GPU OOM / 서버 freeze 가능.

### 구현: deque 기반 커스텀 큐 + 단일 worker

- `analysis_queue: deque` + `analysis_queue_lock: asyncio.Lock()` 추가
- `analysis_worker()` 함수: 큐에서 순서대로 꺼내 `process_analysis()` 호출, GPU 동시 접근 없음
- `lifespan`에서 `asyncio.create_task(analysis_worker())`로 상시 실행, 종료 시 `cancel()`

### 수정: `ai_server.py` (v2.8 → v2.9)

| 변경 | 내용 |
|------|------|
| 임포트 추가 | `asyncio`, `collections.deque` |
| `FastAPI` 임포트에서 제거 | `BackgroundTasks` (더 이상 사용 안 함) |
| 글로벌 변수 추가 | `analysis_queue`, `analysis_queue_lock`, `analysis_worker_task` |
| 신규 함수 | `analysis_worker()` — 단일 worker 루프 |
| `/analyze` 수정 | `background_tasks.add_task()` → `analysis_queue.append()` |
| `/analysis/cancel` 수정 | `cancelled_diary_ids.add()` + 큐에서 해당 diary_id 즉시 제거 |
| `lifespan` 수정 | worker 태스크 시작/종료 관리 |

---

## v2.8 — 2026-04-04 Graceful 모델 로드 처리

> **API 변경 없음** — 에러 처리 내부 로직만 변경. 백엔드 연동 스펙 그대로.

### 배경
`analyzer.py`의 `initialize()`에 예외 처리 없음 → 모델 파일 없으면 서버 크래시.

### 수정: `analyzer.py`

`initialize()`의 각 단계(KURE / Stage1 / Stage2)를 개별 `try/except`로 감싸고 명확한 `RuntimeError` raise.
- KURE: `Exception` → `RuntimeError("KURE 임베딩 모델 로드 실패: ...")`
- Stage1/2: `FileNotFoundError` → `RuntimeError("... 모델 파일 없음: 경로")`
- Stage1/2: 기타 `Exception` → `RuntimeError("... 모델 로드 실패: ...")`

### 수정: `ai_server.py` (v2.7 → v2.8)

`lifespan()`에서 `analyzer.initialize()` 실패 시 서버 크래시 대신 경고 출력 후 계속 기동.
- 모델 로드 실패 → `_initialized=False` 상태 유지 → `/analyze` 요청 시 `MODEL_NOT_LOADED` 에러 콜백 전송

---

## v2.7 — 2026-04-04 분석 취소 엔드포인트 추가

### 배경
백엔드 API 명세 추가: 일기 삭제 시 AI 서버에 분석 취소 신호 전송 (`POST /analysis/cancel/{diary_id}`)

### 구현 방식: 취소 플래그 (옵션 A)
- 글로벌 `cancelled_diary_ids: set` 관리
- `process_analysis` 내 3개 체크포인트에서 플래그 확인 후 조기 종료
  - 시작 전 (모델 로드 확인 전)
  - 분석 후 (감정 일치도 검사 전)
  - 피드백 생성 후 (활동 추천 전)
- 콜백 미전송 (취소된 일기 결과는 백엔드에 보내지 않음)

### 수정: `ai_server.py` (v2.6 → v2.7)

| 변경 | 내용 |
|------|------|
| 글로벌 변수 추가 | `cancelled_diary_ids: set = set()` |
| 신규 엔드포인트 | `POST /analysis/cancel/{diary_id}` — 취소 플래그 등록, 항상 200 반환 |
| `/analyze` 수정 | 재분석 요청 시 `cancelled_diary_ids.discard()` 로 이전 취소 상태 초기화 |
| `process_analysis` 수정 | 3개 체크포인트에서 취소 여부 확인 후 조기 return |

### 백엔드 연동 스펙
- 호출 시점: 사용자가 일기 삭제 시 백엔드가 백그라운드로 자동 전송
- 이미 완료됐거나 없는 diary_id → 무시 (에러 없음)
- 응답: `{"message": "Analysis cancelled or ignored"}` (200 OK)

---

실제로 코드에 반영된 내용만 기록합니다.
팀 논의 중인 제안 사항은 `NOTEBOOK_CHANGES.md` 참고.

## 버전 넘버링 규칙 (2026-03-11~)
- 헤더 형식: `## vX.X — YYYY-MM-DD`
- 날짜 병기는 오늘(2026-03-11)부터 적용. 이전 항목은 소급 미적용.
- 코드 변경이 없는 피드백/결정 기록도 동일 체계로 관리.

---

## 수정된 파일

### `KURE_Burnout_2Stage_v3.ipynb`

**경로 불일치 버그 수정** — Colab에서 `FileNotFoundError` 발생하던 문제

```
수정 전: DATA_PATH = "/content/drive/MyDrive/Burnout"
         모든 파일을 루트에서 로드/저장

수정 후: DATA_PATH      = "/content/drive/MyDrive/Burnout"
         DATASET_PATH   = f"{DATA_PATH}/dataset"
         PROCESSED_PATH = f"{DATASET_PATH}/processed"
```

| 셀 | 변경 내용 |
|----|-----------|
| `cell-4` | `DATASET_PATH`, `PROCESSED_PATH` 변수 추가 |
| `cell-6` | Drive 마운트 후 세 경로 존재 여부 출력 |
| `cell-8` | 감성대화 v2 CSV 로드 경로 → `{PROCESSED_PATH}/burnout_train_v2.csv` |
| `cell-9` | 웰니스 Excel 경로 → `{DATASET_PATH}/웰니스 대화 스크립트 데이터셋/웰니스_대화_스크립트_데이터셋.xlsx` |
| `cell-10` | 연속적 대화 Excel 경로 → `{DATASET_PATH}/한국어 감정 정보가 포함된 연속적 대화 데이터셋/한국어_연속적_대화_데이터셋.xlsx` |
| `cell-10` | 감정 레이블 노이즈 정규화 추가 (`'분ㄴ'→'분노'`, `'ㅈ중립'→'중립'` 등) |
| `cell-12` | CSV 저장 경로 → `{PROCESSED_PATH}/stage1_train_v3.csv` 외 3개 + `os.makedirs` 추가 |

---

### `KURE_Burnout_FineTune_v1.ipynb`

**CSV 로드 경로 불일치 버그 수정** — Section 3에서 ❌ 뜨던 문제

Drive에 이미 존재하는 v3 CSV 위치(`{DATA_PATH}/`)와 노트북이 찾는 위치(`{DATA_PATH}/processed/`)가 달랐던 문제.

| 셀 | 변경 내용 |
|----|-----------|
| `cell-ft-6` | CSV 존재 확인 경로: `{DATA_PATH}/processed/{f}` → `{DATA_PATH}/{f}` |
| `cell-ft-8` | CSV 로드 경로: `{DATA_PATH}/processed/stage1_train_v3.csv` → `{DATA_PATH}/stage1_train_v3.csv` (외 3개 동일) |
| `cell-ft-25` | 그래프 저장 경로: `{DATA_PATH}/processed/training_curves_v3_ft.png` → `{DATA_PATH}/training_curves_v3_ft.png` |
| `cell-ft-29` | 모델 저장 경로: `{DATA_PATH}/processed/stage1_model_v3_ft.pt` → `{DATA_PATH}/stage1_model_v3_ft.pt` (외 1개 동일) |

---

### `README.md`

**모델 학습 노트북 섹션 추가**

- Drive 폴더 구조
- 노트북 개요 및 예상 소요시간
- v3 데이터셋 구성 및 목표 성능
- 학습 완료 후 서버 적용 방법

---

### `KURE_Burnout_FineTune_v2.ipynb` (신규 생성)

**v1 학습 실패 분석 후 재설계한 Stage 2 전용 파인튜닝 노트북**

#### v1 실패 원인
| 원인 | 내용 |
|------|------|
| Stage 1 클래스 불균형 | 부정 86.3% vs 긍정 13.7% + Focal Loss 미사용 → 다수 클래스 수렴 |
| dropout 과다 | 0.5 → BERT 파인튜닝 부적합, 그래디언트 소실 |
| backbone 공유 | Stage 1/2가 동일 객체 공유 → Stage 1 실패가 Stage 2에 연쇄 |

#### v2 주요 변경사항

| 항목 | v1 | v2 |
|------|----|----|
| Stage 1 파인튜닝 | 시도 (→ 실패) | 제거 (v3 frozen 유지, F1=0.9877) |
| backbone 공유 | Stage 1/2 공유 | Stage 2 전용 독립 backbone |
| dropout | 0.5 | 0.2 (BERT 권장값) |
| classifier 초기화 | random | v3 Stage 2 가중치 warm-start |
| lr_head | 3e-4 | 1e-4 (warm-start 시 작게) |
| Focal Loss | Stage 1 미사용 | 항상 적용 (gamma=2.0) |

#### 저장 경로
- 모델: `/content/drive/MyDrive/Burnout/dataset/stage2_model_v4.pt`
- 그래프: `/content/drive/MyDrive/Burnout/dataset/training_curves_v4.png`
- Stage 1은 기존 `stage1_model_v3.pt` 그대로 유지

---

---

## 2026-02-27 추가 변경사항

### 사용자 설문 피드백 수집 시스템 (v2.4)

팀 논의 완료 후 반영. 분석 결과를 본 사용자가 AI 판단의 정오를 평가하는 설문을 제출하면,
백엔드가 2주마다 배치로 AI 서버에 전송하여 재학습 방향 도출에 활용합니다.

#### 확정된 데이터 흐름
```
프론트  →  백엔드 (설문 저장 → FeedbackSurveys 테이블)
                 ↓ 2주마다
           AI 서버 POST /feedback/batch
                 ↓
           feedback_data.csv 누적
```

#### 신규 파일: `feedback_store.py`
- `FeedbackStore` 클래스: 배치 피드백 CSV 저장 (스레드 안전)
- `save_batch()`: 2주치 레코드 일괄 저장
- `get_stats()`: 누적 통계 (정확도, 만족도, 카테고리별 오답 분포)
- CSV 컬럼: `received_at`, `period_start`, `period_end`, `predicted_mbi_category`, `is_correct`, `satisfaction_score`, `user_mbi_category`

#### 수정: `models.py`
| 추가 모델 | 설명 |
|-----------|------|
| `FeedbackRecord` | 단일 피드백 레코드 (배치 1행) |
| `FeedbackBatchRequest` | 백엔드 → AI 서버 2주 배치 요청 |
| `FeedbackBatchResponse` | 배치 수신 응답 (수신 건수, 누적 통계) |

#### 수정: `ai_server.py` (v2.3 → v2.4)
| 엔드포인트 | 설명 |
|-----------|------|
| `POST /feedback/batch` | 백엔드 2주 배치 수신, 유효성 검증 후 CSV 저장 |
| `GET /feedback/stats` | 누적 피드백 통계 + 카테고리별 오답 현황 조회 |

#### 수정: `config.py`
- `FEEDBACK_CSV_PATH` 추가 (기본값: `feedback_data.csv`, 환경변수 오버라이드 가능)
- `BATCH_RETRAIN_THRESHOLD` 추가 (기본값: `50`건)

#### 백엔드 연동 스펙 (`POST /feedback/batch`)
```json
{
  "period_start": "2026-02-01",
  "period_end":   "2026-02-14",
  "records": [
    {
      "predicted_mbi_category": "정서적_고갈",
      "is_correct": false,
      "satisfaction_score": 3,
      "user_mbi_category": "자기비하"
    }
  ]
}
```
> `predicted_mbi_category` / `user_mbi_category` 값: **한국어 확정** (`NORMAL` / `정서적_고갈` / `좌절_압박` / `부정적_대인관계` / `자기비하`)

#### 백엔드 신규 테이블 제안: `FeedbackSurveys`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `survey_id` | INT PK | 고유 ID |
| `diary_id` | INT FK → Diaries | 대상 일기 |
| `is_correct` | Boolean | AI 정오 확인 |
| `satisfaction_score` | INT 1~5 | 만족도 |
| `user_mbi_category` | String nullable | 사용자 선택 카테고리 |
| `created_at` | DateTime | 제출 시각 |

---

---

## 2026-02-27 추가 변경사항 (v2.5)

### recommendations 형식 + mbi_category 한국어 통일

백엔드 확인 완료 후 반영. 백엔드 DB가 AI 서버 전송 형식에 맞춰 변경하기로 결정.

#### 수정: `constants.py`
- `ACTIVITY_ATTRIBUTES` 딕셔너리 추가 (45개 활동 전체)
  - `act_category`: `REST` / `VENTILATION` / `SMALL_WIN`
  - `is_active`: 신체 활동 여부
  - `is_outdoor`: 야외 활동 여부
  - `is_social`: 사회적 교류 여부

#### 수정: `models.py`
- `RecommendationItem` 필드 교체

| 이전 | 이후 |
|------|------|
| `activity_id: int` | `act_content: str` |
| `ai_message: str` | `act_category: str` |
| | `is_active: bool` |
| | `is_outdoor: bool` |
| | `is_social: bool` |

#### 수정: `ai_server.py` (v2.4 → v2.5)
- `ACTIVITY_ATTRIBUTES` import 추가
- `generate_recommendations()`: 활동 속성 조회 후 `RecommendationItem` 빌드 방식 변경
- `send_callback()`: recommendations 직렬화 형식 변경

#### 수정: `analyzer.py`
- `mbi_category` 반환값을 영문(`EMOTIONAL_EXHAUSTION` 등) → **한국어** (`정서적_고갈` 등)로 변경
  - 백엔드가 한국어 수신 후 자체 변환하여 DB에 저장

---

### 확정된 콜백 API 명세 (`POST /diaries/analysis-callback`)

AI 서버 → 백엔드 전송 페이로드:

```json
{
  "diary_id": 1,
  "primary_emotion": "부정",
  "primary_score": 0.8732,
  "mbi_category": "정서적_고갈",   // 긍정이면 "NORMAL"
  "emotion_probs": {
    "긍정": 0.1268,
    "부정": 0.8732,
    "statistics": {
      "period": "최근 30일",
      "total_entries": 5,
      "emotion_frequency": {"긍정": 1, "부정": 4},
      "burnout_trend": {"정서적_고갈": 3, "좌절_압박": 1},
      "mbi_distribution": {"정서적_고갈": 0.6, "좌절_압박": 0.2},
      "situation_frequency": {},
      "top_keywords": ["지치", "무기력", "피곤"],
      "insight_messages": ["최근 30일간 부정 감정이 80.0%를 차지합니다."]
    }
  },
  "ai_message": "오늘 많이 지치셨군요...",
  "recommendations": [
    {
      "act_content": "따뜻한 차/코코아 한 잔 마시기",
      "act_category": "REST",
      "is_active": false,
      "is_outdoor": false,
      "is_social": false,
      "ai_message": "지금처럼 지칠 때일수록 잠깐 몸을 따뜻하게 해주는 게 도움돼요."
    },
    {
      "act_content": "일어나자마자 이불 개기",
      "act_category": "SMALL_WIN",
      "is_active": true,
      "is_outdoor": false,
      "is_social": false,
      "ai_message": "작은 것 하나 완성하는 것만으로도 하루가 달라질 수 있어요."
    }
  ]
}
```

#### mbi_category 값 목록
| 값 | 의미 |
|----|------|
| `NORMAL` | 긍정 (번아웃 없음) |
| `정서적_고갈` | Emotional Exhaustion |
| `좌절_압박` | Frustration / Pressure |
| `부정적_대인관계` | Negative Relationship |
| `자기비하` | Self-Deprecation |

#### act_category 값 목록
| 값 | 활동 ID 범위 | 설명 |
|----|-------------|------|
| `REST` | 1~15 | 휴식·이완 활동 |
| `VENTILATION` | 16~30 | 발산·해소 활동 |
| `SMALL_WIN` | 31~45 | 작은 성취 활동 |

#### ai_message (recommendations 내)
| 조건 | 값 |
|------|-----|
| `USE_LLM=true` | LLM이 사용자 텍스트·번아웃 카테고리 기반으로 개인화된 멘트 생성 |
| `USE_LLM=false` (기본) | `""` (빈 문자열) |

> ⚠️ `statistics` 필드는 일기 3개 이상일 때만 포함 (`MIN_DIARY_COUNT_FOR_INSIGHT` 설정)
> ⚠️ `recommendations`는 일기 3개 이상일 때만 포함 (`MIN_DIARY_COUNT_FOR_RECOMMENDATION` 설정)
> ⚠️ `recommendations[].ai_message`는 LLM 비활성화 시 빈 문자열 — 프론트에서 빈 문자열 처리 필요

---

---

## 2026-02-28 변경사항 (v2.6)

### recommendations[].ai_message 복원

#### 수정: `models.py`
- `RecommendationItem`에 `ai_message: str = ""` 필드 추가

#### 수정: `ai_server.py` (v2.5 → v2.6)
- `generate_recommendations()`: `USE_LLM=true` 시 활동별 개인화 멘트 생성
- `send_callback()`: `recommendations[].ai_message` 페이로드에 포함

| 조건 | 값 |
|------|-----|
| `USE_LLM=true` | 활동명 + 번아웃 카테고리 기반 LLM 개인화 멘트 |
| `USE_LLM=false` (기본) | `""` (빈 문자열) |

---

## 2026-02-28 변경사항 (v2.7)

### 피드백 API 재설계

백엔드 실제 제공 형식에 맞게 전면 재설계.

#### 수정: `models.py`

| 클래스 | 변경 내용 |
|--------|----------|
| `FeedbackRecord` | `is_correct` + `satisfaction_score` → `ai_message_rating` + `mbi_category_rating` (각 1~5) |
| `FeedbackBatchRequest` | `period_start/end` + `records` → `feedbacks` |
| `FeedbackBatchResponse` | `model_accuracy` + `category_corrections` → `avg_ai_message_rating` + `avg_mbi_category_rating` + `low_mbi_by_category` |

#### 수정: `feedback_store.py`
- CSV 헤더: `received_at`, `predicted_mbi_category`, `ai_message_rating`, `mbi_category_rating` (4컬럼)
- `save_batch()`: period 파라미터 제거, 배치별 평균 평점 반환
- `get_stats()`: 평균 평점 + `mbi_category_rating <= 2` 카테고리별 집계

#### 수정: `ai_server.py`
- `/feedback/batch` 엔드포인트 검증 로직 업데이트
- 수신 시각 서버에서 자동 기록 (백엔드가 날짜 전송 불필요)

#### 확정된 요청 형식
```json
{
  "feedbacks": [
    {
      "predicted_mbi_category": "정서적_고갈",
      "ai_message_rating": 5,
      "mbi_category_rating": 2
    }
  ]
}
```

---

### emotion_probs Stage 2 확률 추가

#### 수정: `analyzer.py`
- `emotion_probs`에 Stage 2 카테고리 확률 항상 포함
- `primary_emotion == "긍정"` (Stage 2 미실행) → 카테고리 값 `-1.0` 센티널
- `primary_emotion == "부정"` (Stage 2 실행) → 실제 확률 (0.0 ~ 1.0)

```json
// 긍정
"emotion_probs": { "긍정": 0.88, "부정": 0.12,
  "정서적_고갈": -1.0, "좌절_압박": -1.0, "부정적_대인관계": -1.0, "자기비하": -1.0 }

// 부정
"emotion_probs": { "긍정": 0.09, "부정": 0.91,
  "정서적_고갈": 0.61, "좌절_압박": 0.21, "부정적_대인관계": 0.10, "자기비하": 0.08 }
```

> `-1.0` = Stage 2 미계산 센티널. `0.0`과 구분하기 위해 사용.

---

## 2026-03-04 변경사항 (v2.8)

### EEVE 기반 일기체 스타일 트랜스퍼 노트북 추가

#### 배경
Stage 2 FineTune_v4 F1 **0.4839** 정체 원인: 학습 데이터(AI Hub 구어체) vs 실사용자 입력(일기체) 도메인 미스매치.
LLM API 사용 불가(비용·라이선스) → 로컬 오픈소스 EEVE-Korean-Instruct-10.8B (Apache 2.0) 활용.

#### 신규 파일: `notebooks/KURE_Burnout_StyleTransfer_v1.ipynb`

| 섹션 | 내용 |
|------|------|
| 1 | 환경 설정 (bitsandbytes, transformers) |
| 2 | 경로 설정 |
| 3 | EEVE-Korean-Instruct-10.8B 4-bit 양자화 로드 (Colab T4 가능) |
| 4 | 프롬프트 설계 + 소량 테스트 (5건 품질 확인) |
| 5 | 전체 변환 (50건마다 `st_checkpoint.json` 저장, Colab 단절 대비) |
| 6 | 품질 필터링 (최소 10글자, ok/fallback만 유지) |
| 7 | EEVE 언로드 + KURE 로드 (VRAM 확보) |
| 8 | Stage 2 재학습 데이터 준비 (원본+일기체 1:1 혼합) |
| 9 | Stage 2 재학습 (Focal Loss, warm-start from `stage2_model_v3.pt`) |
| 10 | 성능 비교 — v3(0.4754) vs FineTune_v4(0.4839) vs StyleTransfer |
| 11 | 직접 문장 테스트 — 하드코딩 일기체 문장으로 즉시 검증 (**※ 아직 main 미반영**) |

> **⚠️ 섹션 11 누락**: PR #9 머지 시점 문제로 직접 문장 테스트 셀이 main에 포함되지 않음.
> `feat/style-transfer-notebook` 브랜치의 `481bdf5` 커밋을 cherry-pick 또는 재작업 필요.

#### 저장 경로
- 모델: `stage2_model_st.pt`
- 데이터: `diary_train_v1.csv`
- `data_version: 'style_transfer_v1'`

---

### 파일 구조 정리 (PR #10)

```
루트 → notebooks/  : 모든 .ipynb 학습 노트북 9개
루트 → docs/       : API_SPEC.md, CHANGES.md, DEPLOYMENT.md,
                      NOTEBOOK_CHANGES.md, openapi_callback_schema.yaml
```

#### docs/API_SPEC.md 수정
- MBI 코드표 `NONE` → `NORMAL` 오타 수정 (실제 코드값과 일치)

---

## 2026-03-09 변경사항 (v2.9)

### StyleTransfer v2 실험 결과 + 노트북 개선

#### 신규 파일: `notebooks/KURE_Burnout_StyleTransfer_v2.ipynb`

v1(EEVE-Korean-Instruct-10.8B)의 A100 사용량 제한 + T4 OOM 문제로 모델 교체.

| 항목 | v1 | v2 |
|------|----|----|
| LLM | EEVE-Korean-Instruct-10.8B | Qwen2.5-7B-Instruct |
| GPU 요구사항 | A100 (40GB) | T4 (15GB) 가능 |
| max_new_tokens | 150 | 100 |
| 변환 대상 | 전체 39,547건 | 클래스당 1,000건 × 4 = 4,000건 (층화 샘플링) |
| 중국어 fallback | 없음 | 있음 (re 정규식 감지) |
| 체크포인트 | `st_checkpoint.json` | `st_checkpoint_v2.json` |
| 저장 모델 | `stage2_model_st.pt` | `stage2_model_st_v2.pt` |

#### 실험 결과

| 모델 | F1 (macro) |
|------|------------|
| Stage 2 v3 (기준) | 0.4754 |
| Stage 2 FineTune v4 | 0.4839 |
| Stage 2 StyleTransfer v2 | **0.4690** |

- 변환 품질: ok 4,000건 / fallback 0건 / error 0건
- **결론**: 스타일 트랜스퍼가 오히려 성능 저하. 도메인 미스매치가 F1 정체의 주 원인이 아닐 가능성 높음.
- 현재 최고 성능 모델: `stage2_model_v3_ft.pt` (FineTune v4, F1=0.4839)

---

## 변경되지 않은 것

- `feedback.py`, `emotion_match.py`, `insight.py` 등 — 미변경
- 학습 로직, 모델 아키텍처, 하이퍼파라미터 — 미변경

---

## v4.2 — 2026-03-26 generate_diary_data.py 개선 + 데이터 증량 계획

### 수정: `scripts/generate_diary_data.py`

#### 문제 발견: v1 비율 실험 전부 동일 데이터

합성 데이터가 4,026건(카테고리당 ~1,006건)이라 모든 비율(1:3 / 1:1 / 3:1)에서 가용 최대치(1,006건)가 동일하게 사용됨.
즉 v1의 F1 차이(0.4770 / 0.4826 / 0.4835)는 비율 효과가 아닌 학습 노이즈.
비율 실험이 유효하려면 카테고리당 최소 3,300건(1:3 기준) 이상 필요.

#### 스크립트 개선 내역

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| CategoryConsistencyChecker | 기본 활성 (`--skip-consistency`로 비활성) | **기본 비활성** (`--use-consistency`로 활성) |
| 중복 제거 | 없음 | **`seen_texts` set + 기존 CSV 로드** |
| DiaryStyleScorer 정규화 기준 | 40자 | **60자** (80~120자 문장 불이익 완화) |
| `--batch-size` 기본값 | 5 | **10** |
| `--target` 기본값 | 1,000 | **3,000** |
| `--output` 기본 경로 | `diary_synthetic.csv` (CWD) | `dataset/diary_synthetic.csv` |
| `--checkpoint` 기본 경로 | `diary_gen_checkpoint.json` | `dataset/diary_gen_checkpoint.json` |
| `--stage2-model` 기본값 | `stage2_model.pt` | `models/stage2_model_v3.pt` |

#### CategoryConsistencyChecker 비활성 이유
Stage 2 모델 정확도 ~47% → `pred == expected` 필터가 올바른 문장도 53% 확률로 탈락시키는 역효과.
이 필터가 v1 당시 통과율 극도 저하의 주요 원인.

### 신규 파일: `notebooks/KURE_Burnout_SyntheticData_v2.ipynb`

v1 실험 구조를 기반으로, **5:1 / 7:1 고비율** 탐색 노트북.
단, 현재 합성 데이터 부족으로 5:1/7:1도 실질적으로 동일 데이터 → 데이터 증량 후 실행 예정.

| 항목 | 내용 |
|------|------|
| 실험 비율 | 합성:원본 = 5:1 / 7:1 |
| v1 결과 포함 | 비교표에 Synthetic v1 (1:3/1:1/3:1) 수치 기재 |
| 저장 위치 | `models/stage2_model_syn_5to1.pt`, `7to1.pt` |

### 데이터 증량 계획

| 목표량/카테고리 | 총 합성 | 유효해지는 비율 |
|--------------|---------|---------------|
| **6,000건** (목표) | ~24,000건 | 1:3 완전 유효, 1:1 61% 근사 |

```bash
python scripts/generate_diary_data.py --target 6000
```

---

## v4.1 — 2026-03-26 합성 데이터 혼합 학습 실험 결과

코드 변경 없음. 실험 수치 기록.

### 전체 성능 비교

| 모델 | F1 (macro) | v3 대비 |
|------|------------|--------|
| Stage 2 v3 (기준) | 0.4754 | — |
| FineTune v4 (현재 운영) | 0.4839 | +0.0085 |
| E2E v1 (레이어 2, 원본만) | 0.4835 | +0.0081 |
| StyleTransfer v2 | 0.4690 | -0.0064 |
| **Synthetic v1 (합성:원본 = 1:3)** | **0.4770** | **+0.0016** |
| **Synthetic v1 (합성:원본 = 1:1)** | **0.4826** | **+0.0072** |
| **Synthetic v1 (합성:원본 = 3:1)** | **0.4835** | **+0.0081** |

### 주요 관찰

- 합성 비율 증가 → 단조 성능 향상 (1:3 < 1:1 < 3:1) — 포화점 미도달 가능성
- 3:1에서 E2E v1과 동점 (0.4835), FineTune v4 최고치(0.4839) 미달 (차이 0.0004)
- 합성 데이터가 원본 구어체를 효과적으로 보완함은 확인

> ⚠️ **사후 분석 (v4.2에서 정정)**: v1 비율별 F1 차이는 실제로 학습 노이즈.
> 합성 4,026건은 1:3 비율에도 이미 상한 도달 → 1:3/1:1/3:1 모두 동일 데이터로 학습됨.

---

## v4.3 — 2026-03-31 Synthetic v3 실험 결과

코드 변경 없음. 실험 수치 기록.

### 전체 성능 비교

| 모델 | F1 (macro) | v3 대비 |
|------|------------|--------|
| Stage 2 v3 (기준) | 0.4754 | — |
| FineTune v4 (기존 운영) | 0.4839 | +0.0085 |
| E2E v1 (레이어 2, 원본만) | 0.4835 | +0.0081 |
| StyleTransfer v2 | 0.4690 | -0.0064 |
| **Synthetic v3 (합성:원본 = 1:7)** | **0.4849** | **+0.0095 ★** |
| Synthetic v3 (합성:원본 = 1:5) | 0.4796 | +0.0042 |
| Synthetic v3 (합성:원본 = 1:3) | 0.4677 | -0.0077 |
| Synthetic v3 (합성:원본 = 1:1) | 0.4733 | -0.0021 |
| Synthetic v3 (합성:원본 = 3:1) | 0.4590 | -0.0164 |
| Synthetic v3 (합성:원본 = 5:1) | 0.4490 | -0.0264 |
| Synthetic v3 (합성:원본 = 7:1) | 0.4329 | -0.0425 |

### 주요 관찰

- **1:7이 F1 0.4849로 기존 최고치(FineTune v4 0.4839) 돌파** → 새로운 최고
- 합성 비율 증가 → 단조 성능 감소 (원본 비중이 높을수록 유리)
- 합성이 원본을 대체할수록 성능 급락 (7:1: 0.4329, -0.0425)
- 합성 데이터는 소량 보강(1:7~1:5) 용도로만 유효, 원본 대체 불가
- ⚠️ warm-start 모델 없음(Colab 경로 불일치) → random init으로 학습됨. warm-start 적용 시 추가 개선 가능성 있음

### 다음 단계 후보

- warm-start 경로 수정 후 1:7 재실험
- 1:9, 1:11 등 더 낮은 합성 비율 탐색 (포화점 확인)

---

## v4.2 — 2026-03-29 데이터 수치 정정 + 노트북 v3 추가

코드 변경 없음. 수치 정정 및 신규 노트북 기록.

### 원본 데이터셋 실제 수치 확인 (정정)

원본 합계 약 7.5만 건으로 추정했으나, 실제 확인 결과:

| 데이터셋 | 원본 건수 |
|---------|---------|
| 감성대화 (Training) | 51,628건 |
| 감성대화 (Validation) | 6,640건 |
| 웰니스 (02버전) | 19,769건 |
| 웰니스 (기본) | 5,231건 |
| 한국어 감정 연속 대화 | 55,629건 |
| **원본 합계** | **약 139,000건** |
| **전처리 후 (Stage2)** | **43,942건** (생존율 ~32%) |

> 7.5만 건은 과추정이었음. 실제 원본은 약 14만 건이나, 번아웃 4카테고리 조건 필터링 후 4.4만 건이 남음.

### v1 비율 실험 결과 정정

- `mix_datasets()`의 `take_syn = min(목표량, 가용량)` 로직으로 인해
  합성 4,026건(카테고리당 ~1,006건) 상황에서 1:3 이상 모든 비율이 동일 데이터
- v1 비율별 F1 차이(0.4770/0.4826/0.4835)는 학습 노이즈로 재해석
- **비율 실험은 합성 데이터 증량(목표 24,000건) 완료 후 v3 노트북으로 재실시**

### 신규 파일: `notebooks/KURE_Burnout_SyntheticData_v3.ipynb`

합성 24,000건 증량 완료 후 실행할 7개 비율 일괄 실험 노트북.

| 비율 | 카테고리당 필요량 | 유효 여부 (6,000건 기준) |
|------|----------------|----------------------|
| 1:7 | ~1,413건 | ✓ |
| 1:5 | ~1,977건 | ✓ |
| 1:3 | ~3,296건 | ✓ |
| 1:1 | ~9,887건 | 부분 (61%) |
| 3:1 | ~29,661건 | 상한 도달 |
| 5:1 | ~49,435건 | 상한 도달 |
| 7:1 | ~69,209건 | 상한 도달 |

---

## v4.0 — 2026-03-25 합성 데이터 혼합 학습 실험 시작

### 신규 파일: `scripts/generate_diary_data.py`

Ollama 로컬 LLM으로 번아웃 일기체 합성 데이터 생성 스크립트.

| 항목 | 내용 |
|------|------|
| 모델 | Qwen2.5-7B (Ollama, RTX 4060 Ti 16GB) |
| 생성량 | 카테고리별 1,000건 × 4 = **총 4,026건** |
| 필터 | 기본 후처리 → DiaryStyleScorer → 카테고리 일관성 검사 |
| 출력 | `dataset/diary_synthetic.csv` |
| 체크포인트 | 50건마다 `diary_gen_checkpoint.json` |

주요 CLI 옵션: `--model`, `--target`, `--category`, `--threshold`, `--batch-size`, `--skip-consistency`

### 신규 파일: `notebooks/KURE_Burnout_SyntheticData_v1.ipynb`

합성 데이터 혼합 비율 3가지(1:3 / 1:1 / 3:1) 순차 실험 노트북.
E2E v1 구조(KURE 상위 2레이어 해제) + warm-start from `stage2_model_v3.pt`.

| 항목 | 내용 |
|------|------|
| 아키텍처 | E2E v1 동일 (레이어 2 해제, fp16, gradient checkpointing) |
| 실험 비율 | 합성:원본 = 1:3 / 1:1 / 3:1 |
| 저장 위치 | `models/stage2_model_syn_{ratio}.pt` |
| 체크포인트 | `checkpoints/ckpt_epoch_syn_{ratio}.pt` |

### 폴더 구조 추가

```
llm/
├── models/       # 학습 완료 모델 저장 (신규)
└── checkpoints/  # 에폭 체크포인트 (신규)
```

### 수정: `requirements.txt`
- `torch>=2.0.0` 주석 처리 — `pip install -r` 시 PyPI CPU 버전 덮어쓰기 방지

---

## v3.2 — 2026-03-11 E2E v1/v2 실험 결과 기록

코드 변경 없음. 실험 수치 기록.

### 전체 성능 비교

| 모델 | F1 (macro) | v3 대비 |
|------|------------|--------|
| Stage 2 v3 (기준) | 0.4754 | — |
| FineTune v4 (현재 운영) | 0.4839 | +0.0085 |
| E2E v1 (레이어 2 해제) | 0.4835 | +0.0081 |
| E2E v2 (레이어 4 해제) | 0.4721 | -0.0033 |
| StyleTransfer v2 | 0.4690 | -0.0064 |

### 해석
- 레이어를 많이 풀수록 오히려 성능 하락 — KURE 백본을 건드릴수록 범용 임베딩 품질이 저하됨
- E2E 방향은 레이어 2가 최적, 추가 실험 불필요
- 결국 데이터 문제로 귀결 → Ollama 합성 데이터 생성이 다음 단계

### 다음 방향
- Ollama 로컬 합성 데이터 생성 (`scripts/generate_diary_data.py`)
- 합성 데이터 + E2E v1 구조(레이어 2) 결합 실험

---

## v3.3 — 2026-03-11 LLM 피드백 후처리 파이프라인 고도화

### 설계 의도
KoAlpaca-5.8B는 instruction following이 불안정해서 페르소나 무시, 이모지 남발,
프롬프트 누출 같은 엣지케이스가 런타임에서 실제로 발생함.
파인튜닝이 모델 자체의 출력 품질을 높이는 역할이라면,
후처리는 그 출력의 일관성과 안정성을 런타임에서 보장하는 역할.
두 레이어가 함께 있어야 서비스 품질이 보장됨.

> **교수님 설명 포인트**: "LLM 단독 사용의 불안정성을 보완하기 위해
> 룰 기반 후처리 파이프라인을 설계해서 서비스 품질을 보장했다"

### 수정 파일

**`feedback.py`**
- `CATEGORY_CONTEXT` import 추가
- `MIN_FEEDBACK_LEN = 10`, `MAX_FEEDBACK_LEN = 150` 상수 추가
- `_postprocess()` 길이 검사 강화
  - 상한(150자) 초과 시 마지막 완전한 문장까지만 잘라내기
  - 하한/상한 모두 `None` 반환으로 fallback 트리거
- `_check_repetition()` 메서드 추가
  - 4-gram 기반 반복 표현 감지
  - 고유 n-gram 비율 50% 미만이면 탈락
  - 8단어 미만 짧은 문장은 검사 생략
- `_check_category_relevance()` 메서드 추가
  - `CATEGORY_CONTEXT`의 `avoid` 키워드가 피드백에 포함되면 탈락
  - `core_feeling` 키워드 미포함은 경고만 (fallback 과다 방지)
- `_inject_keyword()` 메서드 추가
  - 사용자 키워드가 피드백에 미반영 시 첫 문장 끝에 자연스럽게 삽입
  - 이미 포함된 경우 패스
- `_generate_llm()` 파이프라인 순서 확정
  1. `_postprocess()` — 누출/이모지/길이/불완전 문장
  2. `_check_repetition()` — 반복 표현
  3. `_validate_persona()` — 페르소나 톤
  4. `_check_category_relevance()` — 카테고리 관련성
  5. `_inject_keyword()` — 키워드 반영 (검증 통과 후 적용)

---

## v3.1 — 2026-03-11 LLM 피드백 후처리 & 페르소나 검증

### 수정 파일

**`prompts.py`**
- `PERSONA_VALIDATION` 딕셔너리 추가 — 페르소나별 required/forbidden 패턴 정의

| 페르소나 | required (예시) | forbidden (예시) |
|----------|----------------|------------------|
| 따뜻한 상담사 | 요, 네요, 죠 | 야, ㅋㅋ |
| 실용적 조언자 | 요, 보세요 | ㅋㅋ, 에휴 |
| 친근한 친구 | 야, 해, ㅎㅎ | 습니다, 드립니다 |
| 차분한 멘토 | 요, 어요, 네요 | 야, !, ㅋㅋ |
| 밝은 응원단 | 요, 어요, ! | 야, 습니다 |

**`feedback.py`**
- `import re`, `Optional` 추가
- `PERSONA_VALIDATION` import 추가
- `_postprocess()` 메서드 추가
  - 프롬프트 지시어 누출 제거 (`### 응답`, `규칙:` 등)
  - 이모지 제거 (LLM이 프롬프트 규칙 무시하는 경우 대비)
  - `\n\n` 이후 잘라내기
  - 10자 미만 → `None`
  - 불완전 문장(요/다/죠/어/!/? 로 안 끝나면) → `None`
- `_validate_persona()` 메서드 추가
  - required 패턴 하나 이상 포함 AND forbidden 패턴 없어야 통과
- `_generate_llm()` 개선
  - 최대 2회 시도 루프로 변경
  - 각 시도마다 `_postprocess()` → `_validate_persona()` 순서로 검증
  - 재시도 시 temperature 0.1 상향 (다양성 확보)
  - 2회 모두 실패 시 템플릿 fallback

---

## v3.0 — 2026-03-11 주제 발표 피드백

코드 변경 없음. 발표 후 청중 및 교수님 피드백 기록.

### 질문 목록

| # | 출처 | 내용 |
|---|------|------|
| 1 | 청중 | 사용자 일기를 AI 분석하는데 개인정보 동의를 따로 받으시나요? |
| 2 | 청중 | 부정적 감정이 있으면 무조건 번아웃이라고 판단하나요? |
| 3 | 청중 | 7페이지 퍼센트가 극단적인데, MBI 지수를 보고 작성한 건가요? |
| 4 | 청중 | 부정적·추상적 내용을 어떻게 번아웃으로 판단하나요? 해결 방안은? |
| 5 | 청중 | 번아웃 관련 뉴스 기사나 현실적인 근거가 있나요? |
| 6 | 청중 | 번아웃 카테고리에 솔루션이 어떻게 매칭되어 제공되나요? |
| 7 | 청중 | 긍정적인 일기 내용에서 사용자 행위를 추출해 솔루션으로 제공할 생각이 있나요? |
| 8 | 청중 | 카테고리가 임의 분류인가요? 실질적 근거가 있나요? |
| 9 | 교수님 | 일기보다 챗봇(대화) 형태로 번아웃 진단·솔루션을 제공하는 것이 더 낫지 않나요? |
| 10 | 교수님 | 번아웃 상태에서 일기 쓰기 힘들 텐데, 일기 형식에 회의적입니다. |
| 11 | 교수님 | KoAlpaca를 쓰는 건가요, 파인튜닝을 하는 건가요? 큰 컨트리뷰션이 뭔가요? |
| 12 | 교수님 | 이 프로젝트가 어떤 면에서 기여하는지 더 명확하게 규정하길 원합니다. |

### 대응 필요 사항

**🔴 HIGH — 교수님 지적**

- **[Q11, Q12] Contribution 불명확**: KoAlpaca 미사용 상태임에도 제안서에 기재됨. 핵심 기여를 한 문장으로 재정의 필요. 제안서 정정 필요.
- **[Q9, Q10] 일기 형식 타당성**: 교수님이 챗봇 전환을 직접 제안. 일기 형식 유지 시 근거 보강 또는 챗봇 전환 여부 팀 논의 필요. **→ 미결정**

**🟡 MEDIUM — 발표 완성도**

- **[Q1] 개인정보 동의**: 온보딩 동의 플로우 미설계. 다음 발표 자료에 반영 필요.
- **[Q2, Q4] 번아웃 판단 기준**: Stage 1(긍/부정) → Stage 2(번아웃 카테고리) 2단계 구조로 답변 가능. 발표 자료에 흐름 명시 필요.
- **[Q3] MBI vs 4카테고리 불일치**: 제안서의 MBI 3단계(EE/DP/PA) vs 코드의 4카테고리 독립 분류 괴리. **→ 미결정 (기존 미해결 사항)**
- **[Q6, Q8] 카테고리-솔루션 매핑 근거**: 박수정(2018) 기반이나 발표에서 약하게 전달됨. 매핑 테이블 또는 출처 명시 필요.

**🟢 LOW — 기능 아이디어**

- **[Q7] 긍정 일기 → 개인화 솔루션**: 사용자 본인의 긍정 경험 추출 활용. 현재 MVP 범위 밖, 향후 고려.

---

## v3.4 — 2026-03-23 합성 일기체 데이터 생성 완료

코드 변경 없음. 데이터 생성 완료 기록.

### 생성 결과

| 카테고리 | 생성 건수 |
|----------|----------|
| 정서적_고갈 | 1,000 |
| 좌절_압박 | 1,000 |
| 부정적_대인관계 | 1,000 |
| 자기비하 | 1,000 |
| **합계** | **4,000** |

- **생성 모델**: Qwen2.5-7B (Ollama 로컬, RTX 4060 Ti 16GB)
- **생성 스크립트**: `scripts/generate_diary_data.py`
- **출력 파일**: `diary_synthetic.csv`
- **필터 파이프라인**: 기본 후처리 → DiaryStyleScorer → 카테고리 일관성 검사

### 다음 단계
- Colab에서 합성 데이터 혼합 재학습 (혼합 비율 실험: 합성:원본 = 1:3 / 1:1 / 3:1)
- E2E v1 구조(레이어 2 해제) 결합 여부 결정은 혼합 재학습 결과 확인 후
