# -*- coding: utf-8 -*-
"""
KURE 기반 한국형 번아웃 감정 분류 모델 - Google Colab 버전
- KURE 임베딩 레이어 + 분류 헤드 구조
- 4가지 번아웃 카테고리: 정서적_고갈, 좌절_압박, 부정적_대인관계, 자기비하

사용법:
1. 이 파일을 Colab에 업로드하거나 직접 복사
2. 셀 실행
"""

# ============================================
# 0. Colab 환경 설정 및 패키지 설치
# ============================================

# !pip install -q sentence-transformers transformers torch pandas scikit-learn tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm
import json
import os
import re

# GPU 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================
# 1. 번아웃 카테고리 정의
# ============================================
BURNOUT_CATEGORIES = {
    0: "정서적_고갈",      # Emotional Exhaustion
    1: "좌절_압박",        # Frustration/Pressure  
    2: "부정적_대인관계",   # Negative Interpersonal Relations
    3: "자기비하"          # Self-deprecation
}

# 감정 키워드 → 번아웃 카테고리 매핑
EMOTION_TO_BURNOUT = {
    # 정서적 고갈 (지치고 무기력해요)
    "스트레스": 0, "무력함": 0, "공허함": 0, "답답함": 0, 
    "부담": 0, "지루함": 0, "귀찮음": 0, "후회": 0, "외로움": 0,
    "피곤": 0, "지침": 0, "무기력": 0, "우울": 0, "슬픔": 0,
    
    # 좌절/압박 (답답하고 화나요)
    "억울": 1, "불만": 1, "원망": 1, "분노": 1, "불평": 1,
    "불편함": 1, "불쾌": 1, "질투": 1, "당황": 1, "화남": 1, "짜증": 1,
    
    # 부정적 대인관계
    "갈등": 2, "배신": 2, "무시": 2, "소외": 2, "서운함": 2,
    
    # 자기비하 (불안하고 걱정돼요)
    "불안": 3, "걱정": 3, "초조": 3, "자괴감": 3, "죄책감": 3,
    "혼란함": 3, "두려움": 3, "부끄러움": 3, "자책": 3,
}


# ============================================
# 2. KURE 기반 임베딩 레이어
# ============================================
class KUREEmbeddingLayer(nn.Module):
    """
    KURE 모델을 활용한 임베딩 레이어
    - nlpai-lab/KURE-v1: 1024차원 임베딩
    """
    
    def __init__(
        self, 
        model_name: str = "nlpai-lab/KURE-v1",
        freeze_encoder: bool = True,
        device: str = None
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # KURE 모델 로드
        print(f"Loading KURE model: {model_name}")
        print("(첫 실행 시 모델 다운로드에 몇 분 소요될 수 있습니다)")
        
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name, device=self.device)
        
        # 임베딩 차원
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # 인코더 프리징
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ KURE encoder frozen")
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """텍스트 → KURE 임베딩"""
        embeddings = self.encoder.encode(
            texts, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """배치 인코딩 (대용량 데이터용)"""
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        return embeddings


# ============================================
# 3. 분류 헤드
# ============================================
class BurnoutClassificationHead(nn.Module):
    """번아웃 분류 헤드"""
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


# ============================================
# 4. 전체 모델
# ============================================
class KUREBurnoutClassifier(nn.Module):
    """KURE 임베딩 + 분류 헤드"""
    
    def __init__(
        self,
        model_name: str = "nlpai-lab/KURE-v1",
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
        device: str = None
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # KURE 임베딩 레이어
        self.embedding_layer = KUREEmbeddingLayer(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            device=self.device
        )
        
        # 분류 헤드
        self.classification_head = BurnoutClassificationHead(
            input_dim=self.embedding_layer.embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        self.num_classes = num_classes
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.embedding_layer(texts)
        logits = self.classification_head(embeddings)
        return logits
    
    def predict(self, texts: List[str]) -> Tuple[List[int], List[str], torch.Tensor]:
        """예측"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            probabilities = F.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(logits, dim=-1).cpu().tolist()
            predicted_categories = [BURNOUT_CATEGORIES[label] for label in predicted_labels]
        
        return predicted_labels, predicted_categories, probabilities
    
    def save_model(self, path: str):
        """모델 저장 (Google Drive 저장 권장)"""
        torch.save({
            'classification_head': self.classification_head.state_dict(),
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_layer.embedding_dim
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.classification_head.load_state_dict(checkpoint['classification_head'])
        print(f"✓ Model loaded from {path}")


# ============================================
# 5. 데이터셋 클래스
# ============================================
class BurnoutDataset(Dataset):
    """번아웃 분류 데이터셋"""
    
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }


def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {'texts': texts, 'labels': labels}


# ============================================
# 6. 학습 함수
# ============================================
def train_model(
    model: KUREBurnoutClassifier,
    train_dataset: BurnoutDataset,
    val_dataset: Optional[BurnoutDataset] = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    device: str = None
):
    """모델 학습"""
    device = device or model.device
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    optimizer = torch.optim.AdamW(
        model.classification_head.parameters(),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    
    # 학습 기록
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            texts = batch['texts']
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        
        # 검증
        val_acc = 0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    texts = batch['texts']
                    labels = batch['labels'].to(device)
                    logits = model(texts)
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_acc = 100 * val_correct / val_total
            history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    return model, history


# ============================================
# 7. Colab 유틸리티 함수들
# ============================================

def mount_google_drive():
    """Google Drive 마운트"""
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted at /content/drive")

def upload_files():
    """로컬 파일 업로드"""
    from google.colab import files
    uploaded = files.upload()
    return list(uploaded.keys())

def download_file(filepath: str):
    """파일 다운로드"""
    from google.colab import files
    files.download(filepath)


# ============================================
# 8. 샘플 데이터 생성 (테스트용)
# ============================================
def create_sample_data():
    """테스트용 샘플 데이터 생성"""
    
    train_texts = [
        # 정서적 고갈 (0)
        "오늘도 야근했다. 너무 지친다.", 
        "아무것도 하기 싫고 무력하다.",
        "매일 반복되는 일상이 지겹다.",
        "에너지가 바닥났다.",
        "쉬어도 피로가 안 풀린다.",
        "일하기 싫은데 해야 한다.",
        "번아웃이 온 것 같다.",
        "몸도 마음도 지쳐있다.",
        
        # 좌절/압박 (1)
        "상사에게 또 깨졌다. 억울하다.",
        "노력해도 인정받지 못한다.",
        "불합리한 지시에 화가 난다.",
        "성과를 가로채여서 분하다.",
        "압박감에 숨이 막힌다.",
        "진짜 짜증나는 하루였다.",
        "왜 나만 이런 일을 당해야 하지.",
        "불공평해서 미칠 것 같다.",
        
        # 부정적 대인관계 (2)
        "동료들이 나를 따돌리는 것 같다.",
        "팀 분위기가 너무 안 좋다.",
        "사람들과 대화하기 싫다.",
        "직장 내 갈등으로 힘들다.",
        "상사와의 관계가 최악이다.",
        "아무도 내 편이 없는 것 같다.",
        "사내 정치에 지쳤다.",
        "동료에게 배신당한 기분이다.",
        
        # 자기비하 (3)
        "나는 능력이 없는 것 같다.",
        "실수할까봐 불안하다.",
        "다른 사람들보다 못한 것 같다.",
        "자신감이 바닥이다.",
        "이런 나 자신이 부끄럽다.",
        "내가 잘할 수 있을까 걱정된다.",
        "자꾸 실수해서 자책하게 된다.",
        "나만 뒤처지는 것 같다.",
    ]
    
    train_labels = [0]*8 + [1]*8 + [2]*8 + [3]*8
    
    return train_texts, train_labels


# ============================================
# 9. 메인 실행
# ============================================
def main():
    """메인 실행 함수"""
    
    print("=" * 60)
    print("KURE 번아웃 감정 분류 모델 - Colab 버전")
    print("=" * 60)
    
    # 1. 모델 초기화
    print("\n[1/4] 모델 초기화...")
    model = KUREBurnoutClassifier(
        model_name="nlpai-lab/KURE-v1",
        hidden_dim=256,
        num_classes=4,
        dropout=0.3,
        freeze_encoder=True
    )
    
    # 2. 샘플 데이터 생성
    print("\n[2/4] 샘플 데이터 생성...")
    train_texts, train_labels = create_sample_data()
    train_dataset = BurnoutDataset(train_texts, train_labels)
    print(f"  - 학습 데이터: {len(train_dataset)}개")
    
    # 3. 학습
    print("\n[3/4] 모델 학습...")
    model, history = train_model(
        model=model,
        train_dataset=train_dataset,
        epochs=5,
        batch_size=8,
        learning_rate=1e-3
    )
    
    # 4. 테스트
    print("\n[4/4] 예측 테스트...")
    test_texts = [
        "오늘 정말 피곤하다. 아무것도 하기 싫어.",
        "상사가 또 무리한 요구를 했다. 열받아.",
        "팀원들이 나를 무시하는 것 같아 속상하다.",
        "발표를 앞두고 너무 불안하다. 잘할 수 있을까.",
    ]
    
    labels, categories, probs = model.predict(test_texts)
    
    print("\n" + "=" * 60)
    print("예측 결과")
    print("=" * 60)
    for text, cat, prob in zip(test_texts, categories, probs):
        print(f"\n입력: {text}")
        print(f"예측: {cat}")
        print(f"확률: {dict(zip(BURNOUT_CATEGORIES.values(), prob.cpu().numpy().round(3)))}")
    
    # 모델 저장 (선택)
    # model.save_model("/content/drive/MyDrive/burnout_model.pt")
    
    print("\n✓ 완료!")
    return model, history


# Colab에서 실행
if __name__ == "__main__":
    model, history = main()
