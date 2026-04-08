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
├── ai_server.py       # FastAPI 메인 (v2.13)
├── analyzer.py        # KURE 임베딩 + 2단계 분류 + analyze_batch()
├── feedback.py        # 템플릿/LLM 피드백 생성
├── emotion_match.py   # 감정 일치도 검사
├── insight.py         # 통계 인사이트
├── error_codes.py     # 에러 코드 (AI1xxx~AI5xxx)
├── config.py          # 환경변수
├── constants.py       # 상수 (페르소나, 활동 90개 등)
├── models.py          # Pydantic 모델
├── scripts/
│   └── generate_diary_data.py   # 합성 일기체 데이터 생성 (완료)
├── models/                      # 학습 완료 모델 저장
├── checkpoints/                 # 에폭 체크포인트 (학습 재개용)
├── dataset/
│   ├── stage2_train_v3.csv      # 원본 구어체 (39,547건)
│   ├── stage2_val_v3.csv        # 검증셋 — 구어체 고정 (4,395건)
│   ├── diary_synthetic.csv      # 합성 일기체 전체 (~28,026건)
│   ├── stage2_train_diary.csv   # 합성 일기체 학습용 (25,223건, 90%)
│   └── stage2_val_diary.csv     # 합성 일기체 검증셋 (2,803건, 10% holdout)
├── docs/
│   ├── CHANGES.md
│   ├── NOTEBOOK_CHANGES.md
│   ├── API_SPEC.md
│   ├── DEPLOYMENT.md
│   └── Claude.md
└── notebooks/
    ├── KURE_Burnout_2Stage_v3.ipynb
    ├── KURE_Burnout_FineTune_v2.ipynb
    ├── KURE_Burnout_StyleTransfer_v2.ipynb
    ├── KURE_Burnout_E2E_v1.ipynb
    ├── KURE_Burnout_SyntheticData_v1.ipynb  # 합성 혼합 실험 (완료)
    ├── KURE_Burnout_SyntheticData_v2.ipynb  # 고비율 실험 (5:1/7:1, 대기 중)
    └── KURE_Burnout_SyntheticData_v3.ipynb  # 7개 비율 일괄 실험 (완료)
```

---

## 모델 파이프라인

```
입력 텍스트
  ↓
[Stage 1] KURE 임베딩 → MLP 헤드 → 긍정 / 부정   (F1 ≈ 0.9877)
  ↓ 부정인 경우만
[Stage 2] KURE 임베딩 → MLP 헤드 → 4개 번아웃 카테고리   (F1 ≈ 0.4849)
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
| `stage2_model_syn_1to7.pt` | Synthetic v3 1:7 **(현재 최고)** | **0.4849** |
| `stage2_model_v3_ft.pt` | FineTune v4 (기존 운영) | 0.4839 |
| `stage2_model_st_v2.pt` | StyleTransfer v2 | 0.4690 (하락, 미사용) |

> ⚠️ `stage2_model_syn_1to7.pt`는 warm-start 없이 random init으로 학습됨 (Colab 경로 불일치).
> warm-start 적용 시 추가 개선 가능성 있음.

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
| 합성 데이터 혼합 v1 (1:3~3:1) | F1 0.4770~0.4835 | 데이터 부족으로 실질 동일 데이터 (학습 노이즈) |
| **합성 데이터 혼합 v3 (1:7)** | **F1 0.4849** | **기존 최고 돌파 ★** |

> ⚠️ 스타일 트랜스퍼(기존 데이터 변환)와 합성 데이터 생성(처음부터 새로 생성)은 다른 접근임.

### Synthetic v3 전체 결과 (합성:원본 비율)

| 비율 | F1 | 비고 |
|------|----|------|
| 1:7 | **0.4849** | ★ 최고 |
| 1:5 | 0.4796 | |
| 1:3 | 0.4677 | |
| 1:1 | 0.4733 | |
| 3:1 | 0.4590 | |
| 5:1 | 0.4490 | |
| 7:1 | 0.4329 | |

→ **합성 비율 증가할수록 성능 급락**. 소량 보강(1:7~1:5)만 유효, 원본 대체 불가.

### 현재 상태 (2026-04-08)
- Synthetic v3 1:7이 현재 최고 (F1=0.4849, warm-start 미적용)
- `stage2_val_diary.csv` 생성 완료 → 이중 검증셋 체계 준비 완료
- 다음: `KURE_Burnout_SyntheticData_v4.ipynb` 작성 후 warm-start 1:7/1:9/1:11 재실험

### 데이터 현황

| 파일 | 건수 | 비고 |
|------|------|------|
| stage2_train_v3.csv | 39,547 | 원본 구어체 학습셋 |
| stage2_val_v3.csv | 4,395 | 구어체 검증셋 (고정) |
| diary_synthetic.csv | 28,026 | 합성 일기체 전체 |
| stage2_train_diary.csv | 25,223 | 합성 일기체 학습용 |
| stage2_val_diary.csv | 2,803 | 합성 일기체 검증셋 (10% holdout) ✅ |

### 원본 데이터셋 수치

| 데이터셋 | 원본 건수 |
|---------|---------|
| 감성대화 (Train+Val) | 58,268건 |
| 웰니스 (두 버전 합산) | 25,000건 |
| 한국어 감정 연속 대화 | 55,629건 |
| **원본 합계** | **약 139,000건** |
| **전처리 후 (Stage2 전체)** | **43,942건** (생존율 ~32%) |

### Contribution 프레이밍
- "공개 일기체 데이터 부재 환경에서 LLM 합성 + 일기체 검증 필터 + KURE 파인튜닝을 결합한 도메인 적응 파이프라인 제안"
- 교수님 키워드: **도메인 적응(Domain Adaptation)**

---

## 검증셋 전략

### 현재 문제
검증셋(`stage2_val_v3.csv`)이 순수 구어체 고정이라, 일기체 방향으로 학습된 모델의 실제 성능을 과소평가할 수 있음.

### 채택 전략: 이중 검증셋 운영

| 파일 | 용도 | 구성 |
|------|------|------|
| `stage2_val_v3.csv` | 기존 모델들과 비교용 (구어체 기준) | 원본 구어체 4,395건 고정 |
| `stage2_val_diary.csv` | 실제 사용 환경 근사 평가용 | 합성 일기체 홀드아웃 |

- 두 F1을 함께 리포트
- "구어체 vs 일기체 F1 격차 = 도메인 미스매치의 실험적 근거"로 활용 가능
- `diary_synthetic.csv`에서 10% stratified holdout으로 분리 (random_state=42)
- `stage2_val_diary.csv` 2,803건, `stage2_train_diary.csv` 25,223건 생성 완료

> ✅ 완료 (2026-04-08)

---

## 피드백 생성 파이프라인

- **기본**: `USE_LLM=false` → 템플릿 기반 피드백
- **활성화**: `USE_LLM=true` → KoAlpaca-Polyglot-5.8B 추론

### KoAlpaca 5단계 후처리

| 단계 | 처리 내용 |
|------|---------|
| 1 | 프롬프트 누출 제거 |
| 2 | 4-gram 반복 감지 |
| 3 | 페르소나 톤 검증 |
| 4 | 카테고리 관련성 검사 |
| 5 | 사용자 키워드 주입 |

> "LLM 단독 사용의 불안정성을 보완하기 위해 룰 기반 후처리 파이프라인을 설계해서 서비스 품질을 보장했다"

---

## 주요 상수 / 설정

- **페르소나**: 1~5 (1=따뜻한 상담사, 2=실용적 조언자, 3=친근한 친구, 4=차분한 멘토, 5=밝은 응원단)
- **활동 DB**: 90개 (REST 1~15, 46~60 / VENTILATION 16~30, 61~75 / SMALL_WIN 31~45, 76~90)
- **최소 일기 수**: 3개 미만 시 `mbi_category = NONE`, recommendations 미포함
- **피드백 배치**: 2주마다 백엔드 → AI 서버, `feedback_data.csv` 누적
- **솔루션**: diary당 2개 제공

## 백엔드 keywords 딕셔너리 형식

```json
{
  "나의 유형": "정서적 고갈",
  "감정": "지침",
  "상황/원인": "업무 과다"
}
```

`emotion_match.py`의 `_extract_user_emotion`은 `"나의 유형"` 우선, `"감정"` 폴백 순.

---

## API 안정성

v2.7~v2.13 모든 변경은 **내부 구현 개선**이며 외부 API 계약은 변경 없음.

---

## 주의사항

- `stage1_model.pt` / `stage2_model.pt` 는 llm/ 루트에 위치해야 서버 로드 가능
- KURE 백본은 항상 frozen (E2E 실험 제외)
- `mbi_category` 값은 **한국어**로 통일 (`정서적_고갈` 등), `NORMAL`만 영문 예외
- Stage 2 미실행 시 카테고리 확률은 `-1.0` 센티널 (0.0과 구분)
- 콜백 실패 시 `error_codes.py`의 AI4xxx 계열 에러 확인

---

## 미결정 사항

| 항목 | 내용 |
|------|------|
| MBI 구조 통일 | 제안서의 MBI 3단계(EE/DP/PA) vs 코드의 4카테고리 독립 분류 |
| 일기 vs 챗봇 | 교수님이 챗봇 전환 직접 제안 (2026-03-11 발표 피드백) — 팀 논의 필요 |
| 혼합 비율 최적값 | 현재 1:7 최고(F1=0.4849, warm-start 미적용). SyntheticData v4로 1:7/1:9/1:11 warm-start 재실험 필요 |
| 발표 구조 개편 | 교수님 피드백: "하루치로 번아웃 판단은 의미 없다. 시간축 변화량이 필요하다." 히스토리 누적 → 패턴 변화 → 판단 고도화 흐름을 발표 자료에 명시적으로 표현할 것. insight.py의 burnout_trend/mbi_distribution이 이미 구현돼 있음 — 발표에서 안 보이는 게 문제. |

---

## 아이디어 메모 (우선순위 낮음)

### Discriminator 학습 가능화 (부트스트래핑)
Generator(Ollama LLM, 고정) + Discriminator(학습 가능 분류기) 구조.
룰 기반 초기 데이터 생성 → Discriminator 학습 → 더 정교한 필터로 재생성 → 반복.
진짜 GAN 아님 (Generator 고정). Future work 또는 교수님 상담 시 발전 방향으로 언급하는 정도.

### 검증 데이터 독립성 확보
학습용(Qwen)과 검증용을 다른 모델(ex. llama3)로 생성 → Generator 다변화로 평가 독립성 확보.
합성 데이터 혼합 재학습 완료 후 진행해도 늦지 않음.
