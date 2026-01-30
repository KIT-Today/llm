# 🔥 번아웃 감지 AI 서버 - 배포 및 운영 가이드

## 📋 목차
1. [개요](#개요)
2. [시스템 구조](#시스템-구조)
3. [로컬 실행 방법](#로컬-실행-방법)
4. [서버 배포 방법](#서버-배포-방법)
5. [API 명세](#api-명세)
6. [환경 설정](#환경-설정)
7. [트러블슈팅](#트러블슈팅)

---

## 개요

### 이 서버가 하는 일
```
사용자 일기 텍스트 → 감정 분석 → 번아웃 카테고리 분류 → 맞춤 피드백 생성
```

### 핵심 기능
| 기능 | 설명 |
|------|------|
| 감정 분류 | 긍정/부정 → 4개 번아웃 카테고리 |
| 피드백 생성 | 5가지 페르소나별 맞춤 메시지 |
| 백엔드 연동 | 비동기 분석 + 콜백 전송 |

### 사용 기술
- **프레임워크**: FastAPI (Python)
- **AI 모델**: KURE 임베딩 + PyTorch 분류 모델
- **피드백**: 템플릿 기반 (빠름) / LLM 기반 (선택)

---

## 시스템 구조

### 전체 아키텍처
```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
│   프론트    │ ──▶ │   백엔드    │ ──▶ │     AI 서버         │
│  (React)    │     │  (FastAPI)  │     │  (이 프로젝트)      │
└─────────────┘     └─────────────┘     └─────────────────────┘
                           │                      │
                           ▼                      ▼
                    ┌─────────────┐     ┌─────────────────────┐
                    │ PostgreSQL  │     │  KURE + Stage1/2    │
                    │     DB      │     │     AI 모델         │
                    └─────────────┘     └─────────────────────┘
```

### 분석 흐름
```
1. 사용자가 앱에서 일기 작성
2. 프론트 → 백엔드: 일기 저장 요청
3. 백엔드 → AI 서버: POST /analyze (분석 요청)
4. AI 서버: 즉시 200 OK 반환 (백엔드는 대기 안 함)
5. AI 서버 백그라운드:
   ├── KURE 임베딩 생성
   ├── Stage 1: 긍정/부정 분류
   ├── Stage 2: 번아웃 카테고리 분류 (부정일 때)
   └── 피드백 + 솔루션 추천 생성
6. AI 서버 → 백엔드: POST /diaries/analysis-callback (결과 전송)
7. 백엔드: DB에 분석 결과 저장
8. 앱에서 분석 결과 표시
```

### 파일 구조
```
Burnout/
├── ai_server.py          # 🔥 메인 서버 (FastAPI)
├── prompts.py            # 페르소나 & 프롬프트 관리
├── explainer.py          # XAI 모듈 (판단 근거 설명)
├── stage1_model.pt       # Stage 1 모델 (긍정/부정)
├── stage2_model.pt       # Stage 2 모델 (4개 카테고리)
├── requirements.txt      # 의존성 목록
├── .env.example          # 환경변수 예시
├── README.md             # 프로젝트 설명
└── test_burnout_full.py  # CLI 테스트 도구
```

---

## 로컬 실행 방법

### 1단계: 가상환경 설정
```bash
# 프로젝트 폴더로 이동
cd D:\Programming\Projects\Burnout

# 가상환경 생성 (최초 1회)
python -m venv venv

# 가상환경 활성화
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate
```

### 2단계: 의존성 설치
```bash
pip install -r requirements.txt

# PyTorch GPU 버전 (CUDA 11.8 기준, 선택사항)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3단계: 환경변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집 (필요시)
```

### 4단계: 서버 실행
```bash
# 개발 모드 (코드 변경 시 자동 재시작)
uvicorn ai_server:app --reload --port 8001

# 또는 Python으로 직접 실행
python ai_server.py
```

### 5단계: 테스트
```bash
# 브라우저에서 Swagger UI 열기
http://localhost:8001/docs

# 서버 상태 확인
http://localhost:8001/

# 페르소나 목록 확인
http://localhost:8001/personas
```

---

## 서버 배포 방법

### 옵션 1: 클라우드 VM (AWS EC2, GCP, Naver Cloud 등)

#### 1) 서버 접속
```bash
ssh username@서버IP
```

#### 2) 프로젝트 업로드
```bash
# 방법 A: Git 사용
git clone [레포지토리 URL]
cd Burnout

# 방법 B: 파일 직접 업로드 (scp)
scp -r ./Burnout username@서버IP:/home/username/
```

#### 3) 환경 설정
```bash
# Python 가상환경
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
nano .env  # 백엔드 URL 등 수정
```

#### 4) 프로덕션 실행
```bash
# Gunicorn + Uvicorn (권장)
pip install gunicorn
gunicorn ai_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001

# 또는 Uvicorn 직접 (간단)
uvicorn ai_server:app --host 0.0.0.0 --port 8001 --workers 4
```

#### 5) 백그라운드 실행 (서버 종료해도 유지)
```bash
# 방법 A: nohup
nohup uvicorn ai_server:app --host 0.0.0.0 --port 8001 &

# 방법 B: screen
screen -S ai_server
uvicorn ai_server:app --host 0.0.0.0 --port 8001
# Ctrl+A, D 로 detach

# 방법 C: systemd 서비스 (권장)
# /etc/systemd/system/ai-server.service 파일 생성
```

#### systemd 서비스 파일 예시
```ini
# /etc/systemd/system/ai-server.service
[Unit]
Description=Burnout AI Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Burnout
Environment="PATH=/home/ubuntu/Burnout/venv/bin"
ExecStart=/home/ubuntu/Burnout/venv/bin/uvicorn ai_server:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable ai-server
sudo systemctl start ai-server
sudo systemctl status ai-server  # 상태 확인
```

### 옵션 2: Docker (권장)

#### Dockerfile 생성
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 포트 노출
EXPOSE 8001

# 실행
CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8001"]
```

#### Docker 빌드 및 실행
```bash
# 이미지 빌드
docker build -t burnout-ai-server .

# 컨테이너 실행
docker run -d -p 8001:8001 --name ai-server burnout-ai-server

# 로그 확인
docker logs -f ai-server
```

### 배포 후 확인사항

1. **백엔드 URL 설정**
   ```
   BACKEND_CALLBACK_URL=http://백엔드서버IP:8000/diaries/analysis-callback
   ```

2. **방화벽 설정**
   - 8001 포트 열기
   - 백엔드 서버에서만 접근 허용 (보안)

3. **헬스체크**
   ```bash
   curl http://서버IP:8001/health
   # {"status": "healthy"}
   ```

---

## API 명세

### 기본 정보
- **Base URL**: `http://서버IP:8001`
- **Content-Type**: `application/json`

### 엔드포인트 목록

#### `GET /` - 서버 상태
```json
// Response
{
  "status": "running",
  "service": "Burnout Detection AI Server",
  "device": "cuda",  // 또는 "cpu"
  "model_loaded": true
}
```

#### `GET /health` - 헬스체크
```json
// Response
{"status": "healthy"}
```

#### `POST /analyze` - 분석 요청 (메인 API)
```json
// Request
{
  "diary_id": 105,
  "user_id": 7,
  "history": [
    {
      "diary_id": 105,
      "content": "오늘 상사한테 또 혼났다. 너무 억울하다.",
      "keywords": {"기분": "나쁨"},
      "created_at": "2026-01-30T12:00:00"
    }
  ]
}

// Response (즉시 반환)
{
  "status": "accepted",
  "message": "분석이 시작되었습니다."
}

// 분석 완료 후 백엔드로 콜백 전송됨
```

#### `POST /analyze/sync` - 동기 분석 (테스트용)
```json
// Request: /analyze와 동일

// Response (분석 완료 후 반환)
{
  "diary_id": 105,
  "primary_emotion": "부정",
  "primary_score": 0.92,
  "mbi_category": "FRUSTRATION_PRESSURE",
  "emotion_probs": {"긍정": 0.08, "부정": 0.92},
  "recommendations": [
    {
      "activity_id": 4,
      "ai_message": "억울한 마음이 느껴져요. 잠시 깊게 숨을 쉬어보세요."
    }
  ]
}
```

#### `GET /personas` - 페르소나 목록
```json
// Response
{
  "personas": [
    {"type": "warm_counselor", "name": "따뜻한 상담사", "tone": "부드럽고 다정한"},
    {"type": "practical_advisor", "name": "실용적 조언자", "tone": "차분하고 명확한"},
    {"type": "friendly_buddy", "name": "친근한 친구", "tone": "편하고 친근한"},
    {"type": "calm_mentor", "name": "차분한 멘토", "tone": "담담하고 깊이 있는"},
    {"type": "cheerful_supporter", "name": "밝은 응원단", "tone": "밝고 에너지 넘치는"}
  ]
}
```

#### `POST /test/feedback` - 피드백 테스트
```
// Query Parameters
- category: 정서적_고갈, 좌절_압박, 부정적_대인관계, 자기비하, 긍정
- text: 테스트할 텍스트
- persona: warm_counselor, practical_advisor, friendly_buddy, calm_mentor, cheerful_supporter

// Example
POST /test/feedback?category=좌절_압박&text=상사가화를냈다&persona=friendly_buddy

// Response
{
  "category": "좌절_압박",
  "persona": {
    "type": "friendly_buddy",
    "name": "친근한 친구",
    "tone": "편하고 친근한"
  },
  "feedback": "헐, 진짜 열받았겠다. 나라도 화났을 듯."
}
```

---

## 환경 설정

### .env 파일 설정

```bash
# 백엔드 콜백 URL (필수)
BACKEND_CALLBACK_URL=http://백엔드서버:8000/diaries/analysis-callback

# 모델 파일 경로
MODEL_DIR=.

# LLM 사용 여부 (false: 템플릿, true: KoAlpaca)
USE_LLM=false

# 기본 페르소나
DEFAULT_PERSONA=warm_counselor

# 서버 설정
PORT=8001
HOST=0.0.0.0
```

### 환경별 설정 예시

#### 개발 (로컬)
```bash
BACKEND_CALLBACK_URL=http://127.0.0.1:8000/diaries/analysis-callback
USE_LLM=false
```

#### 스테이징
```bash
BACKEND_CALLBACK_URL=http://staging-backend.example.com/diaries/analysis-callback
USE_LLM=false
```

#### 프로덕션
```bash
BACKEND_CALLBACK_URL=http://api.example.com/diaries/analysis-callback
USE_LLM=true  # GPU 서버인 경우
```

---

## 트러블슈팅

### 자주 발생하는 문제

#### 1. 모듈 import 에러
```
ERROR: Could not import module "ai_server"
```
**해결**: 반드시 `Burnout` 폴더 안에서 실행
```bash
cd D:\Programming\Projects\Burnout
uvicorn ai_server:app --reload --port 8001
```

#### 2. 모델 파일 없음
```
FileNotFoundError: stage1_model.pt
```
**해결**: 모델 파일이 같은 폴더에 있는지 확인
```bash
ls *.pt  # stage1_model.pt, stage2_model.pt 있어야 함
```

#### 3. CUDA 메모리 부족
```
RuntimeError: CUDA out of memory
```
**해결**: CPU 모드로 실행하거나 배치 크기 줄이기
```bash
# CPU 강제 사용
CUDA_VISIBLE_DEVICES="" uvicorn ai_server:app --port 8001
```

#### 4. 콜백 전송 실패
```
❌ 콜백 전송 에러: Connection refused
```
**해결**: 백엔드 서버가 실행 중인지, URL이 맞는지 확인
```bash
curl http://백엔드URL/health  # 백엔드 상태 확인
```

#### 5. PowerShell에서 curl 에러
```
Invoke-WebRequest : 매개 변수 이름 'X'과(와) 일치하는 매개 변수를 찾을 수 없습니다.
```
**해결**: `curl.exe` 사용 또는 브라우저에서 `/docs` 접속
```powershell
curl.exe -X POST "http://localhost:8001/test/feedback"
# 또는 브라우저에서 http://localhost:8001/docs
```

---

## 연락처

- **AI 담당**: 조민성
- **프로젝트**: 한국형 번아웃 감지 앱 (오늘도)

---

*마지막 업데이트: 2026-01-30*
