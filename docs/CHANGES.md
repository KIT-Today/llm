# 변경사항

## v4.4 — 2026-04-08 이중 검증셋 분리 (diary holodout)

> **코드 변경 없음** — 데이터 분리 작업

### 배경
`stage2_val_v3.csv`는 순수 구어체 고정이라 일기체 방향으로 학습된 모델의 실제 성능을 과소평가할 수 있음.
이중 검증셋 전략으로 "구어체 vs 일기체 F1 격차 = 도메인 미스매치의 실험적 근거" 확보.

### 작업 내용

`diary_synthetic.csv` (28,026건) → stratified 10% holdout 분리

| 파일 | 건수 | 용도 |
|------|------|------|
| `stage2_val_diary.csv` | 2,803 | 일기체 검증셋 (홀드아웃, 카테고리당 700건) |
| `stage2_train_diary.csv` | 25,223 | 학습용 합성 일기체 (기존 `diary_synthetic.csv`에서 분리) |

- `random_state=42`, `stratify=category`로 균등 분리
- 컬럼 형식: `text`, `label` (기존 `stage2_val_v3.csv`와 동일)
- 카테고리→label 매핑: `{정서적_고갈:0, 좌절_압박:1, 부정적_대인관계:2, 자기비하:3}`

---

## v2.14 — 2026-04-08 insight.py 카테고리 키 버그 수정

> **API 변경 없음**

### 배경
`analyzer.py`는 v2.5부터 `mbi_category`를 한국어(`"정서적_고갈"` 등)로 반환하는데,
`insight.py`는 영문 키(`"EMOTIONAL_EXHAUSTION"` 등)로 집계하고 있어서
`burnout_trend`가 항상 비어 있고 `MBIAssessment`가 전부 0으로 계산되는 버그.

### 수정: `insight.py`

| 변경 | 내용 |
|------|------|
| `burnout_trend` 집계 | 영문 키 → **한국어 키** (`"정서적_고갈"`, `"좌절_압박"`, `"부정적_대인관계"`, `"자기비하"`) |
| `burnout_trend` 조건 추가 | `mbi_category != "NORMAL"` 체크 추가 (긍정 일기의 NORMAL 카운팅 방지) |
| `_calculate_mbi_assessment` | EE/DP/PA 모든 카테고리 참조 한국어로 교체 |
| `MBIRiskItem.contributing` | 영문 → 한국어 (`["정서적_고갈"]`, `["좌절_압박", "부정적_대인관계"]` 등) |
| `_generate_insight_messages` | 카테고리 라벨 참조 한국어 키 기준으로 수정 |
| `situation_frequency` 집계 | `"상황/원인"` 키 우선 읽도록 수정 (기존은 `"situation"` 키만 읽어 항상 비어 있었음) |

---

## v2.13 — 2026-04-08 히스토리 배치 분석 (임베딩 병렬화)

> **API 변경 없음**

### 배경
`process_analysis`에서 히스토리 일기를 순차적으로 `analyzer.analyze()` 호출 → 일기 N개마다 KURE 임베딩 N회 + GPU 전송 N회 발생. 키워드 5개 이상 시 약 15분 소요.

### 수정: `analyzer.py`

| 추가/변경 | 내용 |
|-----------|------|
| `_get_embeddings_batch(texts)` 추가 | 텍스트 리스트를 `encode()` 1회로 처리, shape `(N, D)` |
| `_prepare_text()` 추가 | text + keywords 전처리 로직 분리 |
| `_empty_result()` 추가 | 빈 텍스트 기본값 반환 로직 분리 |
| `analyze_batch(items)` 추가 | 히스토리 N개 배치 분석 — 임베딩 1회, Stage 1 배치 forward, 부정 샘플만 Stage 2 배치 forward |
| `analyze()` 유지 | 하위 호환 보장 (단건 API 그대로) |

**배치 처리 흐름:**
```
texts N개 → encode() 1회 → Stage 1 forward (N, 2)
→ 부정 인덱스 추출 → Stage 2 forward (K, 4)
→ 결과 조립
```

### 수정: `ai_server.py` (v2.12 → v2.13)

| 변경 | 내용 |
|------|------|
| `process_analysis` | `for diary in history: analyzer.analyze()` 루프 → `analyzer.analyze_batch()` 1회 호출로 교체 |
| 오늘 일기 중복 분석 제거 | 기존: 히스토리 루프 + 오늘 일기 별도 `analyze()` 2회 호출 → 배치 결과 `[0]` 재사용 |
| 배치 실패 fallback | `analyze_batch()` 예외 시 단건 `analyze()` 루프로 자동 폴백 |
| `analyze_sync` 엔드포인트 | 동일하게 `analyze_batch()` 방식으로 교체 |
| 버전 | 2.11.0 → 2.13.0 |

**기대 효과:** 히스토리 N개 기준 임베딩 N회 → 1회. GPU 전송 오버헤드 대폭 감소.

---

## v2.11 — 2026-04-07 활동 DB 45개 → 90개 확장

> **API 변경 없음**

### 수정: `constants.py`

| 카테고리 | 기존 | 추가 | 합계 |
|----------|------|------|------|
| REST | 1~15 | 46~60 | 30개 |
| VENTILATION | 16~30 | 61~75 | 30개 |
| SMALL_WIN | 31~45 | 76~90 | 30개 |
| **전체** | **45개** | **+45개** | **90개** |

- `ACTIVITY_CONTENT`: 46~90 콘텐츠 추가
- `ACTIVITY_ATTRIBUTES`: 46~90 속성 추가 (is_active/is_outdoor/is_social)
- `ACTIVITY_CATEGORY_IDS`: 각 카테고리에 신규 ID 추가

---

## v2.12 — 2026-04-07 사용자 성향 기반 활동 필터링
`generate_recommendations()`가 번아웃 카테고리 → 활동 카테고리 매핑 후 `random.choice()`로 랜덤 선택.
사용자가 "혼자 있고 싶다"고 했는데 "친구에게 전화해 하소연하기"가 추천되는 등 성향 미반영 문제.

### 구현: 키워드 기반 성향 추론 + 속성 필터링 (방식 B)

**성향 추론**: `user_text` + `keywords`에서 매칭
- 비선호 신호 감지 → `False` (해당 속성 활동 제외)
- 선호 신호 감지 → `True` (현재 필터링엔 영향 없음, 향후 가중치용 예약)
- 신호 없음 → `None` (제약 없음)

**폴백**: 필터링 후 후보 0개이면 전체 풀에서 랜덤 선택 (기존 동작 유지)

### 수정: `constants.py`

- `PREFERENCE_SIGNAL_KEYWORDS` 딕셔너리 추가
  - `is_active`: prefer = ["산책", "운동", "뛰", "움직", "몸 쓰"], avoid = ["눕고", "쉬고", "아무것도 하기 싫"]
  - `is_outdoor`: prefer = ["밖", "나가고", "공원", "바깥", "환기"], avoid = ["나가기 싫", "집에", "방에", "밖은"]
  - `is_social`: prefer = ["친구", "누군가", "같이", "사람이 보고"], avoid = ["혼자", "사람 싫", "아무도", "연락 끊"]

### 수정: `ai_server.py` (v2.9 → v2.11, v2.10은 ai_server 미수정)

| 변경 | 내용 |
|------|------|
| 임포트 추가 | `PREFERENCE_SIGNAL_KEYWORDS` |
| 신규 함수 | `infer_user_preference(user_text, keywords)` — 성향 딕셔너리 반환 |
| 신규 함수 | `matches_preference(attrs, preference)` — 비선호 조건 필터링 |
| `generate_recommendations` 수정 | `random.choice(activity_ids)` → 성향 필터링 후 `candidates`에서 선택, 후보 없으면 폴백 |

---

## v2.10 — 2026-04-05 emotion_match 버그 수정 + keywords 형식 지원

> **API 변경 없음**

### 배경
백엔드 keywords 딕셔너리 형식이 `{"나의 유형": "...", "감정": "...", "상황/원인": "..."}` 인데,
`_extract_user_emotion`이 `energy_category` / `detail_keywords` 키만 읽어 항상 `"긍정"` 반환
→ 사용자가 부정을 선택해도 "긍정이라고 하셨지만..." hint가 잘못 붙는 버그.

### 수정: `emotion_match.py`

| 변경 | 내용 |
|------|------|
| `_extract_user_emotion` 반환 타입 | `str` → `Optional[str]` |
| 추출 실패 시 반환값 | `"긍정"` → `None` |
| `check_match` | `user_emotion is None`이면 비교 skip, `is_matched=True` 반환 (hint 미생성) |
| 우선순위 추가 | `"나의 유형"` → `"감정"` → 구형 `energy_category/detail_keywords` 순 |

### 수정: `constants.py`

- `USER_TYPE_TO_EMOTION` 추가: "나의 유형" 필드값 → 내부 카테고리 매핑
- `FEELING_TO_EMOTION` 추가: "감정" 필드값 → 내부 카테고리 매핑

### 백엔드 keywords 형식 (확인된 스펙)
```json
{
  "나의 유형": "정서적 고갈",
  "감정": "지침",
  "상황/원인": "업무 과다"
}
```
- "나의 유형" 가능값: 긍정 / 정서적 고갈 / 좌절/압박 / 부정적 대인관계 / 자기 비하
- "감정" 가능값: 기쁨, 즐거움, 행복함, 지침, 방전됨, 답답함, 불안함, 울고 싶음, 짜증남, 다 싫음, 화남, 혼자 있고 싶음, 무감각, 내가 싫음, 무기력, 아무것도 안 함, 의미 없음

---

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

| 셀 | 변경 내용 |
|----|-----------|
| `cell-ft-6` | CSV 존재 확인 경로: `{DATA_PATH}/processed/{f}` → `{DATA_PATH}/{f}` |
| `cell-ft-8` | CSV 로드 경로: `{DATA_PATH}/processed/stage1_train_v3.csv` → `{DATA_PATH}/stage1_train_v3.csv` (외 3개 동일) |
| `cell-ft-25` | 그래프 저장 경로: `{DATA_PATH}/processed/training_curves_v3_ft.png` → `{DATA_PATH}/training_curves_v3_ft.png` |
| `cell-ft-29` | 모델 저장 경로: `{DATA_PATH}/processed/stage1_model_v3_ft.pt` → `{DATA_PATH}/stage1_model_v3_ft.pt` (외 1개 동일) |

---

### `README.md`

**모델 학습 노트북 섹션 추가**

- Drive 폴더 구조 / 노트북 개요 및 예상 소요시간 / v3 데이터셋 구성 및 목표 성능 / 학습 완료 후 서버 적용 방법

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
| Focal Loss | Stage 1 미사용 | 항상 적용 (gamma=2.0) |

---

## 2026-02-27 추가 변경사항

### 사용자 설문 피드백 수집 시스템 (v2.4)

팀 논의 완료 후 반영. 분석 결과를 본 사용자가 AI 판단의 정오를 평가하는 설문을 제출하면,
백엔드가 2주마다 배치로 AI 서버에 전송하여 재학습 방향 도출에 활용합니다.

#### 신규 파일: `feedback_store.py`
- `FeedbackStore` 클래스: 배치 피드백 CSV 저장 (스레드 안전)
- `save_batch()`, `get_stats()` 구현
- CSV 컬럼: `received_at`, `predicted_mbi_category`, `ai_message_rating`, `mbi_category_rating`

#### 수정: `models.py` — `FeedbackRecord`, `FeedbackBatchRequest`, `FeedbackBatchResponse` 추가
#### 수정: `ai_server.py` (v2.3 → v2.4) — `POST /feedback/batch`, `GET /feedback/stats` 추가
#### 수정: `config.py` — `FEEDBACK_CSV_PATH`, `BATCH_RETRAIN_THRESHOLD` 추가

---

## 2026-02-27 추가 변경사항 (v2.5)

### recommendations 형식 + mbi_category 한국어 통일

#### 수정: `constants.py` — `ACTIVITY_ATTRIBUTES` 딕셔너리 추가 (45개 활동 전체)
#### 수정: `models.py` — `RecommendationItem` 필드 교체 (`activity_id/ai_message` → `act_content/act_category/is_active/is_outdoor/is_social`)
#### 수정: `ai_server.py` (v2.4 → v2.5) — `generate_recommendations()`, `send_callback()` 수정
#### 수정: `analyzer.py` — `mbi_category` 반환값 영문 → 한국어

---

## 2026-02-28 변경사항 (v2.6)

### recommendations[].ai_message 복원

#### 수정: `models.py` — `RecommendationItem`에 `ai_message: str = ""` 추가
#### 수정: `ai_server.py` (v2.5 → v2.6) — LLM 활성 시 활동별 개인화 멘트 생성

---

## 2026-02-28 변경사항 (v2.7)

### 피드백 API 재설계 + emotion_probs Stage 2 확률 추가

#### 수정: `models.py` — `FeedbackRecord`: `is_correct/satisfaction_score` → `ai_message_rating/mbi_category_rating`
#### 수정: `feedback_store.py` — CSV 헤더 및 집계 로직 변경
#### 수정: `analyzer.py` — `emotion_probs`에 Stage 2 카테고리 확률 항상 포함 (`-1.0` 센티널)

---

## 2026-03-04 변경사항 (v2.8)

### EEVE/Qwen 기반 스타일 트랜스퍼 노트북 추가 + 파일 구조 정리

#### 신규: `notebooks/KURE_Burnout_StyleTransfer_v1.ipynb` (EEVE, A100 OOM으로 폐기)
#### 신규: `notebooks/KURE_Burnout_StyleTransfer_v2.ipynb` (Qwen, F1 0.4690 → 성능 하락, 폐기)
#### 파일 구조 정리: 노트북 → `notebooks/`, 문서 → `docs/`

---

## v3.0~v3.4 — 2026-03-11~23

### v3.0 — 주제 발표 피드백 기록 (코드 변경 없음)
### v3.1 — LLM 피드백 후처리 & 페르소나 검증 (`feedback.py`, `prompts.py`)
### v3.2 — E2E v1/v2 실험 결과 기록 (코드 변경 없음)
### v3.3 — LLM 피드백 후처리 파이프라인 고도화 (`feedback.py`: 4-gram 반복 감지, 카테고리 관련성 검사, 키워드 주입)
### v3.4 — 합성 일기체 데이터 생성 완료 (코드 변경 없음, 총 4,000건)

---

## v4.0~v4.3 — 2026-03-25~31

### v4.0 — 합성 데이터 혼합 학습 실험 시작 (`scripts/generate_diary_data.py`, `KURE_Burnout_SyntheticData_v1.ipynb`)
### v4.1 — Synthetic v1 실험 결과 기록 (코드 변경 없음)
### v4.2 — `generate_diary_data.py` 개선 + 데이터 수치 정정 (원본 약 14만건, 전처리 후 43,942건)
### v4.3 — Synthetic v3 실험 결과 기록 (코드 변경 없음, 최고: 1:7 F1=0.4849)
