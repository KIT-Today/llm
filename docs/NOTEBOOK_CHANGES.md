# 노트북 변경사항 및 팀 공유 사항

> 작성일: 2026-02-26

---

## 1. 수정된 노트북

### `KURE_Burnout_2Stage_v3.ipynb`

v3 학습 노트북에서 **Google Drive 경로 불일치** 버그를 수정했습니다.

#### 문제
- 데이터 로드/저장 경로가 로컬 폴더 구조와 맞지 않아 Colab에서 `FileNotFoundError` 발생
- 연속적 대화 데이터셋 감정 레이블에 오타 노이즈 존재 (`'분ㄴ'`, `'ㅈ중립'` 등)

#### 수정 내용

| 셀 | 수정 전 | 수정 후 |
|----|---------|---------|
| `cell-4` | `DATA_PATH` 하나만 존재 | `DATASET_PATH`, `PROCESSED_PATH` 변수 추가 |
| `cell-6` | 단일 경로만 확인 | 세 경로 모두 존재 여부 출력 |
| `cell-8` | `{DATA_PATH}/burnout_train_v2.csv` | `{PROCESSED_PATH}/burnout_train_v2.csv` |
| `cell-9` | `{DATA_PATH}/웰니스_대화_스크립트_데이터셋.xlsx` | `{DATASET_PATH}/웰니스 대화 스크립트 데이터셋/웰니스_대화_스크립트_데이터셋.xlsx` |
| `cell-10` | `{DATA_PATH}/한국어_연속적_대화_데이터셋.xlsx` | `{DATASET_PATH}/한국어 감정 정보가 포함된 연속적 대화 데이터셋/한국어_연속적_대화_데이터셋.xlsx` + 노이즈 정규화 추가 |
| `cell-12` | `{DATA_PATH}/stage1_train_v3.csv` | `{PROCESSED_PATH}/stage1_train_v3.csv` + `os.makedirs` 추가 |

#### 감정 레이블 노이즈 정규화 추가
```python
EMOTION_NORMALIZE = {
    '분ㄴ': '분노', '분': '분노',
    'ㅈ중립': '중립', 'ㄴ중립': '중립', '중림': '중립',
}
```

---

### `KURE_Burnout_FineTune_v1.ipynb`

FineTune 노트북에서 **CSV 로드 경로 불일치** 버그를 수정했습니다.

#### 문제
- Drive에 이미 존재하는 v3 CSV 파일을 찾지 못하는 문제
- `{DATA_PATH}/processed/` 서브폴더를 찾았지만 실제 파일은 `{DATA_PATH}/`에 위치

#### 수정 내용

| 셀 | 수정 전 | 수정 후 |
|----|---------|---------|
| `cell-ft-6` | `{DATA_PATH}/processed/{f}` 경로 확인 | `{DATA_PATH}/{f}` 경로 확인 |
| `cell-ft-8` | `{DATA_PATH}/processed/stage1_train_v3.csv` | `{DATA_PATH}/stage1_train_v3.csv` |
| `cell-ft-25` | `{DATA_PATH}/processed/training_curves_v3_ft.png` | `{DATA_PATH}/training_curves_v3_ft.png` |
| `cell-ft-29` | `{DATA_PATH}/processed/stage1_model_v3_ft.pt` | `{DATA_PATH}/stage1_model_v3_ft.pt` |

---

## 2. 현재 모델 학습 현황

- `KURE_Burnout_FineTune_v1.ipynb` 현재 Colab에서 학습 중 (약 2~3시간 소요)
- 학습 완료 후 결과 확인 예정

### 기대 성능 (v3_ft)
| | Stage 1 F1 | Stage 2 F1 | Stage 2 Acc |
|--|-----------|-----------|------------|
| **현재 (v2)** | 0.9877 | 0.4811 | 48.1% |
| **목표 (v3_ft)** | 유지 | 0.55+ | 55%+ |

---

## 3. 팀 공유 아이디어: 피드백 기반 학습

### 배경
사용자 일기 기반 번아웃 분류 모델의 도메인 특화도를 높이기 위해 **사용자 피드백을 학습 신호로 활용**하는 방식을 제안합니다.

### 제안 흐름
```
AI 분석 결과 표시 ("오늘 감정: 좌절_압박")
        ↓
사용자 피드백 수집
  - 별점 (⭐~⭐⭐⭐⭐⭐) 또는 [맞아요 / 좀 달라요]
  - 틀렸을 경우: "실제로는 어떤 감정이었나요?" (선택지 제시)
        ↓
피드백 데이터 축적 → 주기적 분류기 헤드 재학습
```

### 기대 효과
- 실제 일기 도메인 데이터로 모델 점진적 개선
- 사용자 개인 패턴 반영 (장기적)
- 프라이버시: 원본 텍스트 저장 없이 `(ai_label, user_label)` 쌍만 저장

### 필요한 논의
| 파트 | 논의 내용 |
|------|---------|
| **백엔드** | `feedback_logs` 테이블 설계, 피드백 API 엔드포인트 추가, 재학습 트리거 조건 |
| **프론트** | 피드백 UI 위치 및 형태, 사용자 유도 방식, 피드백 노출 타이밍 |
| **전체** | 개인정보 처리 방침 수정 (피드백 수집 동의), 재학습 주기 및 배포 프로세스 |

### 구현 난이도
- 백엔드 DB 테이블 1개 추가
- AI 서버: 재학습 스크립트 (분류기 헤드만, KURE frozen 유지 → 빠름)
- 프론트: 피드백 UI 컴포넌트 추가

> **현재 단계**: 아이디어 검토 중. 팀 회의 후 방향 결정 필요.

---

## 4. 참고: 노트북 실행 순서 (Colab)

```
1. KURE_Burnout_2Stage_v3.ipynb 전체 실행
   → MyDrive/Burnout/dataset/processed/ 에 v3 CSV 4개 생성
   → MyDrive/Burnout/dataset/ 에 stage1/2_model_v3.pt 저장

2. KURE_Burnout_FineTune_v1.ipynb 전체 실행
   → MyDrive/Burnout/dataset/ 에서 v3 CSV 로드
   → MyDrive/Burnout/dataset/ 에 stage1/2_model_v3_ft.pt 저장

3. 결과 비교 후 서버 적용 결정
   → 성능 향상 확인 시: .pt 파일 llm/ 루트에 복사 후 서버 재시작
```
