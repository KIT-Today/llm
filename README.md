# 🔥 한국형 번아웃 감지 AI 서버 v2.3

## 개요
KURE 기반 2단계 번아웃 감정 분류 + 5가지 페르소나 피드백 생성 시스템

### 전체 아키텍처
```
[프론트엔드 앱]
       ↓
[백엔드 서버 (FastAPI)] ←→ [AI 서버 (이 프로젝트)]
       ↓                         ↓
   [PostgreSQL]           [KURE + Stage1/2 + LLM]
```

### 분석 흐름
```
1. 사용자가 일기 작성
2. 백엔드 → AI 서버: POST /analyze (persona 포함)
3. AI 서버: 즉시 200 OK 반환
4. AI 서버 백그라운드:
   ├─ 페르소나 설정
   ├─ 2주치 모든 일기 분석 (diary_analyses)
   ├─ Stage 1: 긍정/부정 분류
   ├─ Stage 2: 번아웃 4개 카테고리 분류
   ├─ 감정 일치도 검사 (emotion_match)
   ├─ 통계 인사이트 생성 (statistics_insight)
   ├─ 피드백 생성 (ai_message) - 페르소나 반영
   └─ 활동 추천 (recommendations) - 일기 3개 이상일 때
5. AI 서버 → 백엔드: POST /diaries/analysis-callback
6. 백엔드: DB 저장 (is_analyzed = true)
```

---

## 주요 기능

### 1. 2단계 감정 분류
```
텍스트 입력
    ↓
[KURE-v1 임베딩] (1024차원)
    ↓
┌───────────────────────┐
│ Stage 1: 긍정 vs 부정 │  88% 정확도
└───────────────────────┘
    ↓ (부정이면)
┌───────────────────────┐
│ Stage 2: 4개 카테고리 │  47.8% 정확도
└───────────────────────┘
```

### 2. 번아웃 카테고리
| 카테고리 | MBI 코드 | 설명 |
|----------|----------|------|
| 정서적_고갈 | EMOTIONAL_EXHAUSTION | 에너지 소진, 피로, 무력감 |
| 좌절_압박 | FRUSTRATION_PRESSURE | 분노, 불만, 억울함 |
| 부정적_대인관계 | NEGATIVE_RELATIONSHIP | 대인 갈등, 소외감 |
| 자기비하 | SELF_DEPRECATION | 불안, 자책, 수치심 |

### 3. 5가지 페르소나
| 페르소나 | 문자열 코드 | 숫자 코드 | 톤 |
|----------|------------|----------|-----|
| 따뜻한 상담사 | `warm_counselor` | `1` | 부드럽고 다정한 |
| 실용적 조언자 | `practical_advisor` | `2` | 차분하고 명확한 |
| 친근한 친구 | `friendly_buddy` | `3` | 편하고 친근한 |
| 차분한 멘토 | `calm_mentor` | `4` | 담담하고 깊이 있는 |
| 밝은 응원단 | `cheerful_supporter` | `5` | 밝고 에너지 넘치는 |

> **Note:** `persona` 파라미터는 문자열(`"friendly_buddy"`) 또는 숫자(`3`) 모두 지원

### 4. 활동 추천 (act_category)
| 번아웃 카테고리 | 추천 활동 유형 |
|----------------|---------------|
| 정서적_고갈 | REST, SMALL_WIN |
| 좌절_압박 | VENTILATION, REST |
| 부정적_대인관계 | VENTILATION, SMALL_WIN |
| 자기비하 | SMALL_WIN, REST |

### 5. 감정 일치도 검사 (v2.1+)
사용자가 선택한 감정과 AI 분석 결과 비교
- 일치도 점수 (0~1)
- 불일치 시 힌트 메시지 제공

### 6. 통계 인사이트 (v2.1+)
일기 3개 이상 시 자동 생성
- 기간별 감정 빈도
- 반복되는 상황 키워드
- 번아웃 트렌드 분석

---

## 빠른 시작

### 1. 설치
```bash
cd llm

# 가상환경 생성
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 파일 확인
```
llm/
├── stage1_model.pt   # 긍정/부정 분류 모델
├── stage2_model.pt   # 번아웃 카테고리 분류 모델
└── ai_server.py
```

### 3. 서버 실행
```bash
# 템플릿 모드 (기본, 빠름)
uvicorn ai_server:app --reload --port 8001

# LLM 모드 (GPU 필요)
# Windows PowerShell:
$env:USE_LLM="true"; uvicorn ai_server:app --reload --port 8001
# Linux/Mac:
USE_LLM=true uvicorn ai_server:app --reload --port 8001
```

### 4. 테스트
```bash
# 브라우저에서 Swagger UI
http://localhost:8001/docs

# 서버 상태 확인
curl http://localhost:8001/

# 페르소나 목록
curl http://localhost:8001/personas

# 에러 코드 목록
curl http://localhost:8001/errors
```

---

## API 명세

### POST /analyze
일기 분석 요청 (메인 API)

**Request:**
```json
{
  "diary_id": 123,
  "user_id": 7,
  "persona": 3,
  "history": [
    {
      "diary_id": 123,
      "content": "오늘 상사한테 또 혼났다. 너무 억울하다.",
      "keywords": {
        "energy_category": "답답하고 화나요",
        "detail_keywords": ["억울", "분노"],
        "situation": "사람 관계가 힘들어요"
      },
      "created_at": "2026-01-30T18:00:00"
    },
    {
      "diary_id": 122,
      "content": "어제도 야근했다...",
      "keywords": {},
      "created_at": "2026-01-29T22:00:00"
    }
  ]
}
```

**Response:** `200 OK` (즉시 반환)
```json
{
  "status": "accepted",
  "message": "분석이 시작되었습니다."
}
```

### 콜백 응답 (AI → 백엔드)
분석 완료 후 `/diaries/analysis-callback`으로 전송

```json
{
  "diary_id": 123,
  "primary_emotion": "부정",
  "primary_score": 0.92,
  "mbi_category": "FRUSTRATION_PRESSURE",
  "emotion_probs": {
    "긍정": 0.08,
    "부정": 0.92
  },
  "ai_message": "헐, 진짜 열받았겠다. 나라도 화났을 듯.",
  "diary_analyses": [
    {
      "diary_id": 123,
      "primary_emotion": "부정",
      "primary_score": 0.92,
      "mbi_category": "FRUSTRATION_PRESSURE",
      "keywords": ["억울", "화"]
    }
  ],
  "recommendations": [
    {
      "activity_id": 16,
      "ai_message": "헐, 진짜 열받았겠다. '코인 노래방 가서 소리 지르기' 어때?"
    }
  ],
  "emotion_match": {
    "is_matched": true,
    "user_emotion": "좌절_압박",
    "ai_emotion": "좌절_압박",
    "match_score": 1.0,
    "hidden_emotion_hint": null
  },
  "statistics_insight": {
    "period": "weekly",
    "total_entries": 5,
    "emotion_frequency": {"부정": 4, "긍정": 1},
    "situation_frequency": {"사람 관계가 힘들어요": 3},
    "top_keywords": ["억울", "스트레스"],
    "burnout_trend": {"FRUSTRATION_PRESSURE": 3},
    "insight_messages": ["'사람 관계가 힘들어요'을(를) 3번이나 기록하셨네요..."]
  },
  "success": true,
  "errors": [],
  "fallback_used": false
}
```

### POST /analyze/sync
동기 분석 (테스트용) - 콜백 없이 바로 결과 반환

### GET /personas
사용 가능한 페르소나 목록

### POST /config/activities
활동 ID 설정 (백엔드에서 호출)
```json
{
  "REST": [1, 2, 3, 4, 5],
  "VENTILATION": [16, 17, 18, 19, 20],
  "SMALL_WIN": [31, 32, 33, 34, 35]
}
```

### GET /config/activities
현재 설정된 활동 ID 조회

### GET /errors
에러 코드 목록 조회

---

## 에러 코드 체계 (v2.2+)

| 범위 | 카테고리 | 예시 |
|------|----------|------|
| AI1xxx | 모델 관련 | MODEL_NOT_LOADED, INFERENCE_FAILED |
| AI2xxx | 입력 데이터 | INVALID_PERSONA, CONTENT_TOO_SHORT |
| AI3xxx | 분석 처리 | ANALYSIS_FAILED, FEEDBACK_GENERATION_FAILED |
| AI4xxx | 외부 통신 | CALLBACK_FAILED, CALLBACK_TIMEOUT |
| AI5xxx | 시스템 | INTERNAL_ERROR, OUT_OF_MEMORY |

---

## 피드백 생성 모드

### 템플릿 모드 (기본)
- 카테고리 × 페르소나별 사전 정의 템플릿
- 빠름 (~100ms)
- CPU로 실행 가능

### LLM 모드
- KoAlpaca-Polyglot-5.8B (Apache-2.0 라이센스)
- 자연스러운 맞춤형 피드백
- GPU 필요, 느림 (~2-5초)

```bash
# 환경변수로 모드 선택
USE_LLM=false  # 템플릿 (기본)
USE_LLM=true   # LLM
```

---

## 프로젝트 구조 (v2.3)

```
llm/
├── ai_server.py          # 🔥 메인 FastAPI 서버
├── config.py             # 설정
├── constants.py          # 상수 & 매핑 테이블
├── models.py             # Pydantic 모델
├── analyzer.py           # 번아웃 분석 엔진
├── feedback.py           # 피드백 생성기
├── emotion_match.py      # 감정 일치도 검사기
├── insight.py            # 통계 인사이트 생성기
├── error_codes.py        # 에러 코드 정의
├── prompts.py            # 페르소나 & 프롬프트 관리
├── explainer.py          # XAI 모듈 (선택)
│
├── stage1_model.pt       # Stage 1 모델 (긍정/부정)
├── stage2_model.pt       # Stage 2 모델 (4개 카테고리)
│
├── requirements.txt
├── .env.example
├── README.md
├── DEPLOYMENT.md         # 배포 가이드
└── dataset/              # 학습 데이터
```

### 모듈 설명

| 파일 | 역할 |
|------|------|
| `ai_server.py` | FastAPI 앱, 엔드포인트, 백그라운드 처리 |
| `config.py` | 환경변수, 설정값 |
| `constants.py` | 페르소나 매핑, 활동 ID, 감정 키워드 |
| `models.py` | Request/Response Pydantic 모델 |
| `analyzer.py` | KURE 임베딩 + 2단계 분류 모델 |
| `feedback.py` | 템플릿/LLM 피드백 생성 |
| `emotion_match.py` | 사용자 선택 vs AI 분석 비교 |
| `insight.py` | 기간별 통계 인사이트 |
| `error_codes.py` | 에러 코드 & 폴백 처리 |
| `prompts.py` | 5가지 페르소나 정의 & 템플릿 |

---

## 환경변수

```bash
# .env 파일
BACKEND_CALLBACK_URL=http://127.0.0.1:8000/diaries/analysis-callback
MODEL_DIR=.
USE_LLM=false
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 임베딩 | KURE-v1 (1024차원) |
| 분류 모델 | PyTorch (Linear + LayerNorm) |
| LLM | KoAlpaca-Polyglot-5.8B |
| 서버 | FastAPI + uvicorn |
| 데이터셋 | AI Hub 감성대화 말뭉치 |

---

## 버전 히스토리

| 버전 | 날짜 | 변경사항 |
|------|------|----------|
| v2.3 | 2026-02 | 코드 분할 (모듈화) |
| v2.2 | 2026-02 | 에러 코드 체계 추가 |
| v2.1 | 2026-02 | 감정 일치도 검사, 통계 인사이트 추가 |
| v2.0 | 2026-01 | 5가지 페르소나, 백엔드 콜백 구조 |
| v1.0 | 2026-01 | 2단계 분류 모델 |

---

## 참고 문헌
- 박수정 외(2018). 한국형 번아웃 척도 개발
- WHO (2019). Guidelines on mental health at work
- KURE: https://github.com/nlpai-lab/KURE
- KoAlpaca: https://github.com/Beomi/KoAlpaca

---

## 라이센스
- KURE: MIT
- KoAlpaca-Polyglot-5.8B: Apache-2.0
- AI Hub 데이터: 공공 데이터 (출처 표기 필요)
