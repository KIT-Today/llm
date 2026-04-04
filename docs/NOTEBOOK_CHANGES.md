# 노트북 변경사항 및 팀 공유 사항

> 최초 작성: 2026-02-26 / 최종 업데이트: 2026-03-11

---

## 1. 노트북 전체 목록 및 상태

| 노트북 | 설명 | 상태 | 결과 |
|--------|------|------|------|
| `KURE_Burnout_2Stage_v3.ipynb` | 기준 모델 학습 (frozen KURE) | ✅ 완료 | Stage 2 F1 0.4754 |
| `KURE_Burnout_FineTune_v1.ipynb` | 분류기 헤드 파인튜닝 | ✅ 완료 | Stage 2 F1 0.4839 ← 현재 운영 |
| `KURE_Burnout_FineTune_v2.ipynb` | v1 실패 분석 후 재설계 | ✅ 완료 | v1과 동일 구조, v1 결과 채택 |
| `KURE_Burnout_StyleTransfer_v1.ipynb` | EEVE 스타일 트랜스퍼 | ❌ 실패 | A100 OOM |
| `KURE_Burnout_StyleTransfer_v2.ipynb` | Qwen2.5-7B 스타일 트랜스퍼 | ✅ 완료 | Stage 2 F1 0.4690 (하락, 미사용) |
| `KURE_Burnout_E2E_v1.ipynb` | KURE 상위 2레이어 해제 E2E | ✅ 완료 | Stage 2 F1 0.4835 |
| `KURE_Burnout_E2E_v2.ipynb` | KURE 상위 4레이어 해제 E2E | ✅ 완료 | Stage 2 F1 0.4721 (레이어 늘릴수록 하락) |

---

## 2. 전체 성능 비교

| 모델 | F1 (macro) | v3 대비 | 비고 |
|------|------------|---------|------|
| Stage 2 v3 (기준) | 0.4754 | — | frozen KURE |
| FineTune v4 | **0.4839** | +0.0085 | **현재 운영** |
| E2E v1 (레이어 2) | 0.4835 | +0.0081 | FineTune v4와 거의 동일 |
| E2E v2 (레이어 4) | 0.4721 | -0.0033 | 레이어 많이 풀수록 하락 |
| StyleTransfer v2 | 0.4690 | -0.0064 | 미사용 |

### 해석
- E2E는 레이어 2가 최적, 추가 실험 불필요
- 모든 접근이 0.47~0.48 정체 → 근본 원인은 데이터 도메인 미스매치
- **다음 방향**: Ollama 로컬 합성 데이터 생성 + E2E v1 구조 결합

---

## 3. 다음 실험 계획

### Ollama 합성 데이터 생성 + 재학습
- 스크립트: `scripts/generate_diary_data.py` (미구현, Claude.md 참고)
- 로컬(RTX 4060 Ti)에서 생성, Colab에서 학습 — 병렬 진행 가능
- 혼합 비율 실험: 합성:원본 = 1:3 / 1:1 / 3:1
- 모델 구조: E2E v1 (KURE 상위 2레이어 해제) 유지

---

## 4. 피드백 기반 학습 (구현 완료)

> 아이디어 단계에서 `feedback_store.py`로 구현 완료. CHANGES.md v2.4~v2.7 참고.

### 흐름
```
AI 분석 결과 표시
  ↓
사용자 평가 수집 (ai_message_rating 1~5, mbi_category_rating 1~5)
  ↓
2주마다 백엔드 → AI 서버 배치 전송 (POST /feedback/batch)
  ↓
feedback_data.csv 누적 → 재학습 방향 도출
```

### 현재 상태
- `feedback_store.py`: 배치 저장, 통계 집계 구현 완료
- `POST /feedback/batch`, `GET /feedback/stats` 엔드포인트 운영 중
- 재학습 자동화는 미구현 (수동 트리거)

---

## 5. 노트북 실행 순서 (Colab)

```
[기준 데이터 준비]
1. KURE_Burnout_2Stage_v3.ipynb
   → processed/ 에 stage1/2_train/val_v3.csv 생성
   → stage1_model_v3.pt, stage2_model_v3.pt 저장

[현재 운영 모델]
2. KURE_Burnout_FineTune_v1.ipynb
   → stage2_model_v3_ft.pt 저장 (F1 0.4839)
   → llm/ 루트에 복사 후 서버 적용

[실험용 — 미채택]
3. KURE_Burnout_StyleTransfer_v2.ipynb
   → stage2_model_st_v2.pt (F1 0.4690, 미사용)
4. KURE_Burnout_E2E_v1.ipynb
   → stage2_model_e2e_v1.pt (F1 0.4835, 참고용)

[예정]
5. 합성 데이터 생성 후 E2E v1 구조로 재학습
   → scripts/generate_diary_data.py (로컬) → CSV → Colab
```
