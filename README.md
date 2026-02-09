# 🔥 한국형 번아웃 감지 AI 서버 v2.0

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
| 페르소나 | 코드 | 톤 |
|----------|------|-----|
| 따뜻한 상담사 | `warm_counselor` | 부드럽고 다정한 |
| 실용적 조언자 | `practical_advisor` | 차분하고 명확한 |
| 친근한 친구 | `friendly_buddy` | 편하고 친근한 |
| 차분한 멘토 | `calm_mentor` | 담담하고 깊이 있는 |
| 밝은 응원단 | `cheerful_supporter` | 밝고 에너지 넘치는 |

### 4. 활동 추천 (act_category)
| 번아웃 카테고리 | 추천 활동 유형 |
|----------------|---------------|
| 정서적_고갈 | REST, SMALL_WIN |
| 좌절_압박 | VENTILATION, REST |
| 부정적_대인관계 | VENTILATION, SMALL_WIN |
| 자기비하 | SMALL_WIN, REST |

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
  "persona": "friendly_buddy",
  "history": [
    {
      "diary_id": 123,
      "content": "오늘 상사한테 또 혼났다. 너무 억울하다.",
      "keywords": {"기분": "나쁨"},
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
    },
    {
      "diary_id": 122,
      "primary_emotion": "부정",
      "primary_score": 0.78,
      "mbi_category": "EMOTIONAL_EXHAUSTION",
      "keywords": ["야근"]
    }
  ],
  "recommendations": [
    {
      "activity_id": 4,
      "ai_message": "좀 쉬어야 할 것 같아. 산책이라도 하자!"
    }
  ]
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
  "REST": [1, 2, 3],
  "VENTILATION": [4, 5, 6],
  "SMALL_WIN": [7, 8, 9]
}
```

### GET /config/activities
현재 설정된 활동 ID 조회

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

## 프로젝트 구조
```
llm/
├── ai_server.py          # 🔥 메인 FastAPI 서버
├── prompts.py            # 페르소나 & 프롬프트 관리
├── explainer.py          # XAI 모듈
├── stage1_model.pt       # Stage 1 모델
├── stage2_model.pt       # Stage 2 모델
├── requirements.txt
├── .env.example
├── README.md
├── DEPLOYMENT.md         # 배포 가이드
└── MEETING_NOTES.md      # 회의 자료
```

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
