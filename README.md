# 🔥 한국형 번아웃 감지 AI 서버

## 개요
KURE(Korea University Retrieval Embedding) 기반 2단계 번아웃 감정 분류 + 피드백 생성 시스템

### 전체 아키텍처
```
[프론트엔드 앱]
       ↓
[백엔드 서버 (FastAPI)] ←→ [AI 서버 (이 프로젝트)]
       ↓                         ↓
   [PostgreSQL]           [KURE + Stage1/2 모델]
```

### 분석 흐름
```
1. 사용자가 일기 작성
2. 백엔드 → AI 서버: POST /analyze (비동기)
3. AI 서버: 분석 시작, 즉시 200 OK 반환
4. AI 서버 백그라운드:
   ├─ Stage 1: 긍정/부정 분류
   ├─ Stage 2: 번아웃 4개 카테고리 분류 (부정일 때)
   └─ 피드백 + 솔루션 추천 생성
5. AI 서버 → 백엔드: POST /diaries/analysis-callback
6. 백엔드: DB 저장 및 앱에 알림
```

## 모델 구조

### 2단계 분류 시스템
```
텍스트 입력
    ↓
[KURE-v1 임베딩] (1024차원)
    ↓
┌───────────────────────┐
│ Stage 1: 긍정 vs 부정 │ (88% 정확도)
└───────────────────────┘
    ↓ (부정이면)
┌───────────────────────┐
│ Stage 2: 4개 카테고리 │ (47.8% 정확도)
└───────────────────────┘
    ↓
피드백 + 솔루션 추천 생성
```

### 번아웃 카테고리 (박수정 외, 2018)
| 코드 | 카테고리 | MBI 코드 | 설명 |
|------|----------|----------|------|
| 0 | 정서적_고갈 | EMOTIONAL_EXHAUSTION | 에너지 소진, 피로, 무력감 |
| 1 | 좌절_압박 | FRUSTRATION_PRESSURE | 분노, 불만, 억울함 |
| 2 | 부정적_대인관계 | NEGATIVE_RELATIONSHIP | 대인 갈등, 소외감 |
| 3 | 자기비하 | SELF_DEPRECATION | 불안, 자책, 수치심 |

---

## 빠른 시작

### 1. 설치
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# PyTorch GPU 버전 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 환경변수 설정
```bash
cp .env.example .env
# .env 파일 수정
```

### 3. 모델 파일 확인
```
Burnout/
├── stage1_model.pt   # 긍정/부정 분류 모델
├── stage2_model.pt   # 번아웃 카테고리 분류 모델
└── ai_server.py
```

### 4. AI 서버 실행
```bash
# 개발 모드
uvicorn ai_server:app --reload --port 8001

# 프로덕션 모드
uvicorn ai_server:app --host 0.0.0.0 --port 8001 --workers 4
```

### 5. 테스트
```bash
# 서버 상태 확인
curl http://localhost:8001/

# 동기 분석 테스트
curl -X POST "http://localhost:8001/analyze/sync" \
  -H "Content-Type: application/json" \
  -d '{
    "diary_id": 1,
    "user_id": 1,
    "history": [{
      "diary_id": 1,
      "content": "오늘 상사한테 또 혼났다. 너무 억울하고 화가 난다.",
      "keywords": {"기분": "나쁨"},
      "created_at": "2026-01-27T12:00:00"
    }]
  }'
```

---

## API 명세

### POST /analyze
백엔드에서 호출하는 분석 요청 API

**Request:**
```json
{
  "diary_id": 105,
  "user_id": 7,
  "history": [
    {
      "diary_id": 105,
      "content": "오늘 너무 힘들었다...",
      "keywords": {"기분": "나쁨", "사건": "업무과다"},
      "created_at": "2026-01-25T18:00:00"
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
분석 완료 후 백엔드의 `/diaries/analysis-callback`으로 전송

```json
{
  "diary_id": 105,
  "primary_emotion": "부정",
  "primary_score": 0.92,
  "mbi_category": "FRUSTRATION_PRESSURE",
  "emotion_probs": {
    "긍정": 0.08,
    "부정": 0.92
  },
  "recommendations": [
    {
      "activity_id": 4,
      "ai_message": "억울한 마음이 느껴져요. 잠시 깊게 숨을 쉬어보세요."
    },
    {
      "activity_id": 5,
      "ai_message": "스트레스가 많으셨군요. 바람 쐬는 것도 좋아요."
    }
  ]
}
```

### POST /analyze/sync
테스트용 동기 분석 API (결과 직접 반환)

### POST /test/feedback
피드백 생성 테스트

---

## 프로젝트 구조
```
Burnout/
├── ai_server.py                 # 🔥 메인 FastAPI 서버
├── test_burnout_full.py         # CLI 테스트 도구
├── explainer.py                 # XAI 모듈 (Attention, SHAP, 키워드)
├── stage1_model.pt              # Stage 1 모델 가중치
├── stage2_model.pt              # Stage 2 모델 가중치
├── requirements.txt
├── .env.example
├── src/
│   ├── kure_emotion_classifier.py
│   └── data_preprocessing.py
├── dataset/
│   ├── 018.감성대화/
│   └── 한국어 감정 정보가 포함된 연속적 대화 데이터셋/
└── KURE_Burnout_*.ipynb         # Colab 학습 노트북
```

---

## 피드백 생성

### 템플릿 기반 (기본, 빠름)
- 카테고리별 사전 정의된 템플릿에서 랜덤 선택
- 응답 시간: ~100ms

### LLM 기반 (선택, 고품질)
- KoAlpaca-Polyglot-5.8B 사용
- 성격별 프롬프트 (따뜻한 상담사, 실용적 조언자, 친근한 친구)
- GPU 필요, 응답 시간: ~2-5초

```bash
# LLM 모드 활성화
USE_LLM=true uvicorn ai_server:app --reload --port 8001
```

---

## 배포 체크리스트

- [ ] `BACKEND_CALLBACK_URL` 실제 주소로 변경
- [ ] `Config.ACTIVITY_IDS` 실제 DB activity_id로 매핑
- [ ] GPU 서버 확보 (LLM 사용 시)
- [ ] 모델 파일 배포 (stage1_model.pt, stage2_model.pt)
- [ ] CORS 설정 확인 (필요시 제한)
- [ ] 로깅 설정
- [ ] 모니터링 (Prometheus/Grafana 등)

---

## 참고 문헌
- 박수정 외(2018). 한국형 번아웃 척도 개발
- WHO (2019). Guidelines on mental health at work
- KURE: https://github.com/nlpai-lab/KURE

---

## CLI 테스트 도구

```bash
# 샘플 테스트
python test_burnout_full.py

# 단일 텍스트 분석
python test_burnout_full.py --text "오늘 너무 힘들었어"

# 인터랙티브 모드
python test_burnout_full.py -i

# JSON 출력
python test_burnout_full.py --text "화가 난다" --json
```
