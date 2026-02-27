# 변경사항 (2026-02-27 추가 / 2026-02-26 최초)

실제로 코드에 반영된 내용만 기록합니다.
팀 논의 중인 제안 사항은 `NOTEBOOK_CHANGES.md` 참고.

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

## 변경되지 않은 것

- `feedback.py`, `emotion_match.py`, `insight.py` 등 — 미변경
- 학습 로직, 모델 아키텍처, 하이퍼파라미터 — 미변경
