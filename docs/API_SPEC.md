# AI 서버 API 명세

- **Base URL**: `http://ai-server-ip:8001`
- **Content-Type**: `application/json`

---

## 6. AI 서버 (AI Server)

파일 위치: `ai_server.py`
핵심 로직: `analyzer.py`, `feedback.py`, `emotion_match.py`, `insight.py`

> **참고:** 백엔드가 AI 서버를 호출하는 방향과, AI 서버가 백엔드로 결과를 되돌려주는 콜백 방향 두 가지로 통신합니다.

---

### 6-1. 일기 분석 요청

- **API**: `POST /analyze`
- **호출 주체**: 백엔드 → AI 서버
- **작동 로직**:
  1. 백엔드가 최신 일기 + 2주치 히스토리를 함께 전송합니다.
  2. AI 서버는 즉시 `200 OK`를 반환합니다. (백엔드가 기다리지 않음)
  3. 백그라운드에서 분석을 수행하고, 완료되면 `/diaries/analysis-callback`으로 결과를 전송합니다.

- **Request Body**:
```json
{
  "diary_id": 105,
  "user_id": 7,
  "persona": 2,
  "history": [
    {
      "diary_id": 105,
      "content": "오늘 너무 힘들었다...",
      "keywords": {
        "energy_category": "지치고 무기력해요",
        "detail_keywords": ["스트레스", "무력함"],
        "situation": "업무 방식이 문제예요"
      },
      "created_at": "2026-02-20T18:00:00"
    },
    {
      "diary_id": 102,
      "content": "어제는 조금 괜찮았는데",
      "keywords": {},
      "created_at": "2026-02-19T20:00:00"
    }
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `diary_id` | int | 이번에 새로 쓴 일기 ID (결과를 돌려받을 기준) |
| `user_id` | int | 유저 ID |
| `persona` | int 또는 str | 페르소나 번호(1~5) 또는 문자열 코드 |
| `history` | List | 최신 일기 포함 2주치 일기 목록. **첫 번째 항목이 방금 작성한 최신 일기** |

- **Response** (즉시 반환, Code: 200):
```json
{
  "status": "accepted",
  "message": "분석이 시작되었습니다."
}
```

---

### 6-2. 분석 결과 콜백

- **API**: `POST /diaries/analysis-callback`
- **호출 주체**: AI 서버 → 백엔드 (프론트엔드가 직접 호출하지 않음)
- **작동 로직**:
  1. 분석이 완료되면 AI 서버가 백엔드의 이 엔드포인트로 결과를 전송합니다.
  2. 백엔드는 감정 분석 결과(`EmotionAnalysis`)와 추천 솔루션(`SolutionLog`)을 DB에 저장합니다.
  3. 일기가 3개 미만이면 `mbi_category`를 강제로 `"NONE"`으로 처리합니다.
  4. `ai_message`는 일기 개수와 관계없이 항상 저장됩니다.

- **Request Body** (AI → 백엔드):
```json
{
  "diary_id": 105,
  "primary_emotion": "부정",
  "primary_score": 0.91,
  "mbi_category": "정서적_고갈",
  "emotion_probs": {
    "긍정": 0.09,
    "부정": 0.91,
    "정서적_고갈": 0.61,
    "좌절_압박": 0.21,
    "부정적_대인관계": 0.10,
    "자기비하": 0.08,
    "statistics": { ... }
  },
  "ai_message": "많이 지치셨겠어요. 오늘 하루도 정말 수고하셨습니다.",
  "recommendations": [
    {
      "act_content": "따뜻한 차/코코아 한 잔 마시기",
      "act_category": "REST",
      "is_active": false,
      "is_outdoor": false,
      "is_social": false,
      "ai_message": ""
    }
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `diary_id` | int | 분석 대상 일기 ID |
| `primary_emotion` | str | `"긍정"` 또는 `"부정"` |
| `primary_score` | float | 감정 확률 (0.0 ~ 1.0) |
| `mbi_category` | str | `NORMAL` 또는 한국어 카테고리 4종 |
| `emotion_probs` | dict | Stage 1 + Stage 2 확률 항상 포함 (아래 참고) |
| `ai_message` | str | 페르소나가 반영된 AI 총평 메시지 |
| `recommendations` | List | 추천 활동 목록 (일기 3개 이상일 때만 포함) |

#### emotion_probs 구조

항상 6개 키 포함. `primary_emotion == "긍정"`이면 Stage 2 미실행 → 카테고리 값은 `-1.0` 센티널.

```json
// 긍정일 때 (Stage 2 미실행)
"emotion_probs": {
  "긍정": 0.88,
  "부정": 0.12,
  "정서적_고갈": -1.0,
  "좌절_압박": -1.0,
  "부정적_대인관계": -1.0,
  "자기비하": -1.0
}

// 부정일 때 (Stage 2 실행)
"emotion_probs": {
  "긍정": 0.09,
  "부정": 0.91,
  "정서적_고갈": 0.61,
  "좌절_압박": 0.21,
  "부정적_대인관계": 0.10,
  "자기비하": 0.08,
  "statistics": { ... }   // 일기 3개 이상 시만 포함
}
```

> ⚠️ Stage 2 카테고리 값이 `-1.0`이면 미계산 상태. UI에서 표시 시 필터링 필요.

#### recommendations 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `act_content` | str | 활동 내용 |
| `act_category` | str | `REST` / `VENTILATION` / `SMALL_WIN` |
| `is_active` | bool | 신체 활동 여부 |
| `is_outdoor` | bool | 야외 활동 여부 |
| `is_social` | bool | 사회적 교류 여부 |
| `ai_message` | str | 개인화 멘트 (`USE_LLM=true` 시), 기본값 `""` |

- **Response** (Code: 200): `"Analysis & Solutions saved successfully"`

---

### 6-3. 서버 상태 확인

- **API**: `GET /`
- **호출 주체**: 백엔드 또는 운영자

- **Response** (Code: 200):
```json
{
  "status": "running",
  "service": "Burnout Detection AI Server",
  "version": "2.3.0",
  "device": "cpu",
  "model_loaded": true,
  "features": [
    "emotion_analysis",
    "emotion_match_check",
    "statistics_insight",
    "activity_recommendation"
  ]
}
```

---

### 6-4. 헬스체크

- **API**: `GET /health`
- **호출 주체**: 로드밸런서, 모니터링 시스템

- **Response** (Code: 200):
```json
{ "status": "healthy" }
```

---

### 6-5. 페르소나 목록 조회

- **API**: `GET /personas`

- **Response** (Code: 200):
```json
{
  "personas": [
    { "type": "warm_counselor",    "name": "따뜻한 상담사", "tone": "부드럽고 다정한" },
    { "type": "practical_advisor", "name": "실용적 조언자", "tone": "차분하고 명확한" },
    { "type": "friendly_buddy",    "name": "친근한 친구",   "tone": "편하고 친근한" },
    { "type": "calm_mentor",       "name": "차분한 멘토",   "tone": "담담하고 깊이 있는" },
    { "type": "cheerful_supporter","name": "밝은 응원단",   "tone": "밝고 에너지 넘치는" }
  ]
}
```

---

### 6-6. 활동 ID 설정 / 조회

- **API**: `POST /config/activities` / `GET /config/activities`
- **호출 주체**: 백엔드 (DB의 활동 ID 목록을 AI 서버에 동기화할 때 사용)

- **POST Request Body**:
```json
{
  "REST":        [1, 2, 3, 4, 5],
  "VENTILATION": [16, 17, 18, 19, 20],
  "SMALL_WIN":   [31, 32, 33, 34, 35]
}
```

- **GET Response** (Code: 200):
```json
{
  "activities": {
    "REST":        [1, 2, 3, ...],
    "VENTILATION": [16, 17, 18, ...],
    "SMALL_WIN":   [31, 32, 33, ...]
  }
}
```

---

### 6-7. 피드백 배치 수신

- **API**: `POST /feedback/batch`
- **호출 주체**: 백엔드 → AI 서버 (주기적 배치)

- **Request Body**:
```json
{
  "feedbacks": [
    {
      "predicted_mbi_category": "정서적_고갈",
      "ai_message_rating": 5,
      "mbi_category_rating": 2
    },
    {
      "predicted_mbi_category": "자기비하",
      "ai_message_rating": 3,
      "mbi_category_rating": 4
    }
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `predicted_mbi_category` | str | AI가 예측한 카테고리 (한국어) |
| `ai_message_rating` | int 1~5 | AI 총평 메시지 만족도 |
| `mbi_category_rating` | int 1~5 | MBI 카테고리 적절성 평가 |

- **Response** (Code: 200):
```json
{
  "status": "saved",
  "received": 2,
  "total_accumulated": 150,
  "avg_ai_message_rating": 4.0,
  "avg_mbi_category_rating": 3.0,
  "low_mbi_by_category": {
    "정서적_고갈": 3
  }
}
```

> `low_mbi_by_category`: `mbi_category_rating <= 2`인 레코드를 카테고리별로 집계. 모델의 취약 카테고리 파악에 활용.

---

### 6-8. 피드백 통계 조회

- **API**: `GET /feedback/stats`
- **호출 주체**: 운영자

- **Response** (Code: 200):
```json
{
  "total": 150,
  "avg_ai_message_rating": 3.9,
  "avg_mbi_category_rating": 3.4,
  "low_mbi_by_category": {
    "정서적_고갈": 12,
    "자기비하": 7
  }
}
```

---

### @ MBI 카테고리 코드표

| 한국어 | mbi_category 값 | 설명 |
|--------|-----------------|------|
| 긍정 | `NORMAL` | 번아웃 없음 |
| 정서적 고갈 | `EMOTIONAL_EXHAUSTION` | 에너지 소진, 피로, 무력감 |
| 좌절/압박 | `FRUSTRATION_PRESSURE` | 분노, 불만, 억울함 |
| 부정적 대인관계 | `NEGATIVE_RELATIONSHIP` | 대인 갈등, 소외감 |
| 자기비하 | `SELF_DEPRECATION` | 불안, 자책, 수치심 |

### @ 페르소나 코드표

| 숫자 | 문자열 코드 | 이름 |
|------|------------|------|
| 1 | `warm_counselor` | 따뜻한 상담사 |
| 2 | `practical_advisor` | 실용적 조언자 |
| 3 | `friendly_buddy` | 친근한 친구 |
| 4 | `calm_mentor` | 차분한 멘토 |
| 5 | `cheerful_supporter` | 밝은 응원단 |

> `persona` 파라미터는 숫자(`2`) 또는 문자열(`"practical_advisor"`) 모두 허용합니다.
