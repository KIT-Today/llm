# -*- coding: utf-8 -*-
"""
AI Hub 데이터셋 전처리 및 번아웃 카테고리 매핑
- 감성대화 말뭉치 (60개 감정 → 4개 번아웃 카테고리)
- 한국어 감정 정보 대화 데이터셋 (7개 감정)
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re


# ============================================
# 1. 번아웃 카테고리 매핑 테이블
# ============================================

# 4가지 번아웃 카테고리
BURNOUT_LABELS = {
    0: "정서적_고갈",       # Emotional Exhaustion
    1: "좌절_압박",         # Frustration/Pressure
    2: "부정적_대인관계",    # Negative Interpersonal Relations
    3: "자기비하"           # Self-deprecation
}

# AI Hub 감성대화 말뭉치 감정 → 번아웃 매핑
# 박수정 외(2018) 한국형 번아웃 4요인 모델 기반
EMOTION_60_TO_BURNOUT = {
    # ========== 정서적 고갈 (0) ==========
    # 에너지 소진, 피로, 무력감 관련
    "피곤": 0, "지침": 0, "무기력": 0, "공허": 0, "답답": 0,
    "스트레스": 0, "부담": 0, "지루함": 0, "귀찮음": 0,
    "외로움": 0, "우울": 0, "슬픔": 0, "허탈": 0, "힘듦": 0,
    "눈물": 0, "위축": 0, "좌절": 0, "절망": 0, "고독": 0,
    "서러움": 0, "아픔": 0, "괴로움": 0, "후회": 0,
    
    # ========== 좌절/압박 (1) ==========
    # 분노, 불만, 억울함 관련
    "분노": 1, "화남": 1, "짜증": 1, "억울": 1, "불만": 1,
    "원망": 1, "불평": 1, "불쾌": 1, "당황": 1, "질투": 1,
    "증오": 1, "환멸": 1, "혐오": 1, "경멸": 1, "적대감": 1,
    "원한": 1, "배신감": 1, "실망": 1, "좌절감": 1,
    
    # ========== 부정적 대인관계 (2) ==========
    # 대인관계 갈등, 소외감 관련
    "소외감": 2, "무시당함": 2, "배척": 2, "갈등": 2,
    "불신": 2, "의심": 2, "거부감": 2, "적대": 2,
    "섭섭함": 2, "서운함": 2, "오해": 2, "미움": 2,
    
    # ========== 자기비하 (3) ==========
    # 불안, 자책, 수치심 관련
    "불안": 3, "걱정": 3, "초조": 3, "두려움": 3, "공포": 3,
    "자책": 3, "죄책감": 3, "수치심": 3, "부끄러움": 3,
    "자괴감": 3, "열등감": 3, "무능감": 3, "창피": 3,
    "혼란": 3, "당혹": 3, "긴장": 3,
    
    # ========== 긍정 감정 (제외 또는 별도 처리) ==========
    # 번아웃 분석에서는 제외하거나 "정상" 카테고리로
    "기쁨": -1, "행복": -1, "즐거움": -1, "만족": -1,
    "감사": -1, "사랑": -1, "설렘": -1, "희망": -1,
    "뿌듯함": -1, "안도": -1, "편안": -1, "평온": -1,
    "흥분": -1, "감동": -1, "자부심": -1, "성취감": -1,
}


# 한국어 감정 정보 대화 데이터셋 (7개 감정) → 번아웃 매핑
EMOTION_7_TO_BURNOUT = {
    "분노": 1,      # 좌절/압박
    "슬픔": 0,      # 정서적 고갈
    "불안": 3,      # 자기비하 (불안 기반)
    "상처": 2,      # 부정적 대인관계
    "당황": 1,      # 좌절/압박
    "기쁨": -1,     # 제외
    "중립": -1,     # 제외
}


# ============================================
# 2. 텍스트 전처리
# ============================================
class TextPreprocessor:
    """음성 일기 및 대화 텍스트 전처리"""
    
    # 한국어 필러 워드
    FILLER_WORDS = [
        "음", "어", "아", "에", "으", "그", "저", "이",
        "뭐", "막", "좀", "이제", "그래서", "근데", "그냥",
        "진짜", "되게", "엄청", "완전", "약간", "솔직히"
    ]
    
    @staticmethod
    def clean_text(text: str) -> str:
        """기본 텍스트 정제"""
        if not text:
            return ""
        
        # 공백 정규화
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 반복 문자 정규화 (ㅋㅋㅋㅋ → ㅋㅋ)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 특수문자 정리 (기본적인 것만)
        text = re.sub(r'[^\w\s가-힣.,!?]', '', text)
        
        return text
    
    @staticmethod
    def remove_fillers(text: str) -> str:
        """필러 워드 제거 (STT 후처리용)"""
        words = text.split()
        cleaned = [w for w in words if w not in TextPreprocessor.FILLER_WORDS]
        return ' '.join(cleaned)
    
    @staticmethod
    def preprocess_for_model(text: str, remove_filler: bool = False) -> str:
        """모델 입력용 전처리"""
        text = TextPreprocessor.clean_text(text)
        if remove_filler:
            text = TextPreprocessor.remove_fillers(text)
        return text


# ============================================
# 3. AI Hub 데이터 로더
# ============================================
class AIHubDataLoader:
    """AI Hub 데이터셋 로더"""
    
    def __init__(self, base_path: str = "D:/Programming/Projects/Burnout/dataset"):
        self.base_path = Path(base_path)
        self.preprocessor = TextPreprocessor()
    
    def load_emotional_dialogue(self, split: str = "train") -> pd.DataFrame:
        """
        감성대화 말뭉치 로드
        - 경로: 018.감성대화/
        """
        data_path = self.base_path / "018.감성대화"
        
        all_data = []
        
        # JSON 파일들 탐색
        for json_file in data_path.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 구조에 따라 파싱 (AI Hub 형식)
                if isinstance(data, list):
                    for item in data:
                        parsed = self._parse_emotional_dialogue_item(item)
                        if parsed:
                            all_data.extend(parsed)
                elif isinstance(data, dict):
                    parsed = self._parse_emotional_dialogue_item(data)
                    if parsed:
                        all_data.extend(parsed)
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} samples from 감성대화 말뭉치")
        return df
    
    def _parse_emotional_dialogue_item(self, item: dict) -> List[dict]:
        """감성대화 데이터 아이템 파싱"""
        results = []
        
        # AI Hub 감성대화 형식에 따라 조정 필요
        # 일반적인 구조: {"talk": {"content": {...}, "emotion": "..."}}
        try:
            if "talk" in item:
                talk = item["talk"]
                content = talk.get("content", {})
                
                # 발화별로 추출
                for key, value in content.items():
                    if isinstance(value, dict):
                        text = value.get("text", value.get("sentence", ""))
                        emotion = value.get("emotion", talk.get("emotion", ""))
                        
                        if text and emotion:
                            results.append({
                                "text": self.preprocessor.preprocess_for_model(text),
                                "original_emotion": emotion,
                                "source": "감성대화"
                            })
            
            # 다른 형식도 처리
            elif "sentence" in item or "text" in item:
                text = item.get("sentence", item.get("text", ""))
                emotion = item.get("emotion", item.get("label", ""))
                
                if text and emotion:
                    results.append({
                        "text": self.preprocessor.preprocess_for_model(text),
                        "original_emotion": emotion,
                        "source": "감성대화"
                    })
                    
        except Exception as e:
            pass
        
        return results
    
    def load_continuous_dialogue(self, split: str = "train") -> pd.DataFrame:
        """
        한국어 감정 정보가 포함된 연속적 대화 데이터셋 로드
        """
        data_path = self.base_path / "한국어 감정 정보가 포함된 연속적 대화 데이터셋"
        
        all_data = []
        
        for json_file in data_path.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        parsed = self._parse_continuous_dialogue_item(item)
                        if parsed:
                            all_data.extend(parsed)
                elif isinstance(data, dict):
                    # 대화 목록이 있는 경우
                    dialogues = data.get("data", data.get("dialogues", [data]))
                    for item in dialogues:
                        parsed = self._parse_continuous_dialogue_item(item)
                        if parsed:
                            all_data.extend(parsed)
                            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} samples from 연속적 대화 데이터셋")
        return df
    
    def _parse_continuous_dialogue_item(self, item: dict) -> List[dict]:
        """연속적 대화 데이터 아이템 파싱"""
        results = []
        
        try:
            # 대화 형식
            if "utterances" in item:
                for utt in item["utterances"]:
                    text = utt.get("utterance", utt.get("text", ""))
                    emotion = utt.get("emotion", "")
                    
                    if text and emotion:
                        results.append({
                            "text": self.preprocessor.preprocess_for_model(text),
                            "original_emotion": emotion,
                            "source": "연속대화"
                        })
            
            # 단일 문장 형식
            elif "text" in item or "sentence" in item:
                text = item.get("text", item.get("sentence", ""))
                emotion = item.get("emotion", item.get("label", ""))
                
                if text and emotion:
                    results.append({
                        "text": self.preprocessor.preprocess_for_model(text),
                        "original_emotion": emotion,
                        "source": "연속대화"
                    })
                    
        except Exception as e:
            pass
        
        return results


# ============================================
# 4. 번아웃 카테고리 매핑
# ============================================
class BurnoutMapper:
    """감정 레이블 → 번아웃 카테고리 매핑"""
    
    def __init__(self):
        self.emotion_60_map = EMOTION_60_TO_BURNOUT
        self.emotion_7_map = EMOTION_7_TO_BURNOUT
    
    def map_emotion_to_burnout(self, emotion: str) -> int:
        """
        감정 레이블을 번아웃 카테고리로 매핑
        Returns: 0-3 (번아웃 카테고리) 또는 -1 (제외)
        """
        emotion = emotion.strip().lower()
        
        # 60개 감정 매핑 시도
        for key, value in self.emotion_60_map.items():
            if key in emotion or emotion in key:
                return value
        
        # 7개 감정 매핑 시도
        for key, value in self.emotion_7_map.items():
            if key in emotion or emotion in key:
                return value
        
        # 키워드 기반 휴리스틱 매핑
        return self._heuristic_mapping(emotion)
    
    def _heuristic_mapping(self, emotion: str) -> int:
        """키워드 기반 휴리스틱 매핑"""
        
        # 정서적 고갈 키워드
        if any(k in emotion for k in ["지침", "피곤", "무기력", "공허", "우울", "슬픔", "힘"]):
            return 0
        
        # 좌절/압박 키워드
        if any(k in emotion for k in ["분노", "화", "짜증", "억울", "불만", "불쾌"]):
            return 1
        
        # 부정적 대인관계 키워드
        if any(k in emotion for k in ["소외", "무시", "갈등", "배척", "서운"]):
            return 2
        
        # 자기비하 키워드
        if any(k in emotion for k in ["불안", "걱정", "초조", "두려", "자책", "죄책", "부끄"]):
            return 3
        
        # 긍정 감정 키워드 (제외)
        if any(k in emotion for k in ["기쁨", "행복", "즐거", "만족", "감사", "사랑"]):
            return -1
        
        # 매핑 실패
        return -1
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임 전체 매핑 처리"""
        
        df = df.copy()
        df['burnout_label'] = df['original_emotion'].apply(self.map_emotion_to_burnout)
        
        # 제외된 항목(-1) 필터링
        before_count = len(df)
        df = df[df['burnout_label'] != -1]
        after_count = len(df)
        
        print(f"Mapping complete: {after_count}/{before_count} samples retained")
        print(f"  - Excluded (positive/neutral): {before_count - after_count}")
        
        # 분포 출력
        print("\nBurnout category distribution:")
        for label, name in BURNOUT_LABELS.items():
            count = len(df[df['burnout_label'] == label])
            print(f"  {label}. {name}: {count} ({100*count/len(df):.1f}%)")
        
        return df


# ============================================
# 5. 데이터셋 생성 파이프라인
# ============================================
def create_burnout_dataset(
    base_path: str = "D:/Programming/Projects/Burnout/dataset",
    output_path: str = "D:/Programming/Projects/Burnout/processed",
    min_text_length: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    전체 파이프라인: AI Hub 데이터 → 번아웃 분류 데이터셋
    """
    
    # 1. 데이터 로드
    print("=" * 50)
    print("Step 1: Loading AI Hub datasets")
    print("=" * 50)
    
    loader = AIHubDataLoader(base_path)
    
    df_emotional = loader.load_emotional_dialogue()
    df_continuous = loader.load_continuous_dialogue()
    
    # 2. 데이터 병합
    print("\n" + "=" * 50)
    print("Step 2: Merging datasets")
    print("=" * 50)
    
    df_combined = pd.concat([df_emotional, df_continuous], ignore_index=True)
    print(f"Combined dataset size: {len(df_combined)}")
    
    # 3. 번아웃 매핑
    print("\n" + "=" * 50)
    print("Step 3: Mapping to burnout categories")
    print("=" * 50)
    
    mapper = BurnoutMapper()
    df_mapped = mapper.process_dataframe(df_combined)
    
    # 4. 필터링
    print("\n" + "=" * 50)
    print("Step 4: Filtering")
    print("=" * 50)
    
    # 최소 길이 필터
    df_filtered = df_mapped[df_mapped['text'].str.len() >= min_text_length]
    print(f"After length filter: {len(df_filtered)}")
    
    # 중복 제거
    df_filtered = df_filtered.drop_duplicates(subset=['text'])
    print(f"After deduplication: {len(df_filtered)}")
    
    # 5. Train/Val 분할
    print("\n" + "=" * 50)
    print("Step 5: Train/Validation split")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        df_filtered, 
        test_size=0.1, 
        stratify=df_filtered['burnout_label'],
        random_state=42
    )
    
    print(f"Train set: {len(train_df)}")
    print(f"Validation set: {len(val_df)}")
    
    # 6. 저장
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv(output_dir / "val.csv", index=False, encoding='utf-8-sig')
    
    print(f"\nSaved to {output_dir}")
    
    return train_df, val_df


# ============================================
# 6. 메인
# ============================================
if __name__ == "__main__":
    # 전처리 파이프라인 실행
    train_df, val_df = create_burnout_dataset()
    
    print("\n" + "=" * 50)
    print("Sample data:")
    print("=" * 50)
    print(train_df.head(10).to_string())
