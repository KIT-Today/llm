# -*- coding: utf-8 -*-
"""
KURE 기반 한국형 번아웃 감정 분류 모델
- KURE 임베딩 레이어 + 분류 헤드 구조
- 4가지 번아웃 카테고리: 정서적_고갈, 좌절_압박, 부정적_대인관계, 자기비하
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


# ============================================
# 1. 번아웃 카테고리 정의
# ============================================
BURNOUT_CATEGORIES = {
    0: "정서적_고갈",      # Emotional Exhaustion
    1: "좌절_압박",        # Frustration/Pressure  
    2: "부정적_대인관계",   # Negative Interpersonal Relations
    3: "자기비하"          # Self-deprecation
}

# 감정 키워드 → 번아웃 카테고리 매핑 (프로젝트 문서 기반)
EMOTION_TO_BURNOUT = {
    # 정서적 고갈 (지치고 무기력해요)
    "스트레스": 0, "무력함": 0, "공허함": 0, "답답함": 0, 
    "부담": 0, "지루함": 0, "귀찮음": 0, "후회": 0, "외로움": 0,
    
    # 좌절/압박 (답답하고 화나요)
    "억울": 1, "불만": 1, "원망": 1, "분노": 1, "불평": 1,
    "불편함": 1, "불쾌": 1, "질투": 1, "당황": 1,
    
    # 부정적 대인관계 - 상황 기반으로도 매핑
    "갈등": 2, "배신": 2, "무시": 2, "소외": 2,
    
    # 자기비하 (불안하고 걱정돼요)
    "불안": 3, "걱정": 3, "초조": 3, "자괴감": 3, "죄책감": 3,
    "혼란함": 3, "조마조마함": 3, "두려움": 3, "부끄러움": 3,
}


# ============================================
# 2. KURE 기반 임베딩 레이어
# ============================================
class KUREEmbeddingLayer(nn.Module):
    """
    KURE 모델을 활용한 임베딩 레이어
    - nlpai-lab/KURE-v1: 1024차원 임베딩
    - Sentence-Transformers 기반으로 사용 가능
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
        
        # KURE 모델 로드 (Sentence Transformers)
        print(f"Loading KURE model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        
        # 임베딩 차원 (KURE-v1은 1024차원)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # 인코더 프리징 옵션
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("KURE encoder frozen (only classification head will be trained)")
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        텍스트 → KURE 임베딩 변환
        """
        embeddings = self.encoder.encode(
            texts, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """배치 인코딩"""
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        return embeddings


# ============================================
# 3. 분류 헤드 (Classification Head)
# ============================================
class BurnoutClassificationHead(nn.Module):
    """
    번아웃 분류를 위한 분류 헤드
    - KURE 임베딩 → 4개 클래스 분류
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # KURE 임베딩 차원
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
        """임베딩 → 로짓"""
        return self.classifier(embeddings)


# ============================================
# 4. 전체 모델: KURE + 분류 헤드
# ============================================
class KUREBurnoutClassifier(nn.Module):
    """
    KURE 임베딩 + 분류 헤드를 결합한 전체 모델
    """
    
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
        """텍스트 → 분류 로짓"""
        embeddings = self.embedding_layer(texts)
        logits = self.classification_head(embeddings)
        return logits
    
    def predict(self, texts: List[str]) -> Tuple[List[int], List[str], torch.Tensor]:
        """
        예측 수행
        Returns:
            - predicted_labels: 예측된 레이블 인덱스
            - predicted_categories: 예측된 카테고리명
            - probabilities: 각 클래스별 확률
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            probabilities = F.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(logits, dim=-1).cpu().tolist()
            predicted_categories = [BURNOUT_CATEGORIES[label] for label in predicted_labels]
        
        return predicted_labels, predicted_categories, probabilities
    
    def save_model(self, path: str):
        """분류 헤드 저장 (KURE 인코더는 별도 저장 불필요)"""
        torch.save({
            'classification_head': self.classification_head.state_dict(),
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_layer.embedding_dim
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """분류 헤드 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.classification_head.load_state_dict(checkpoint['classification_head'])
        print(f"Model loaded from {path}")


# ============================================
# 5. 데이터셋 클래스
# ============================================
class BurnoutDataset(Dataset):
    """번아웃 분류용 데이터셋"""
    
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
    """배치 콜레이트 함수"""
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
    """
    모델 학습 함수
    """
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
    
    # 분류 헤드만 학습 (KURE 인코더는 frozen)
    optimizer = torch.optim.AdamW(
        model.classification_head.parameters(),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
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
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
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
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    return model


# ============================================
# 7. 테스트 및 사용 예제
# ============================================
def main():
    """테스트 및 사용 예제"""
    
    # 1. 모델 초기화
    print("=" * 50)
    print("KURE 번아웃 분류 모델 초기화")
    print("=" * 50)
    
    model = KUREBurnoutClassifier(
        model_name="nlpai-lab/KURE-v1",
        hidden_dim=256,
        num_classes=4,
        dropout=0.3,
        freeze_encoder=True  # KURE 인코더 고정, 분류 헤드만 학습
    )
    
    # 2. 샘플 데이터로 테스트
    print("\n" + "=" * 50)
    print("샘플 데이터 테스트")
    print("=" * 50)
    
    # 샘플 문장들 (번아웃 관련)
    sample_texts = [
        "오늘도 야근이다. 너무 지치고 힘들다.",  # 정서적 고갈
        "상사가 또 화를 냈다. 억울하고 분하다.",  # 좌절/압박
        "팀원들이 나를 무시하는 것 같다.",  # 부정적 대인관계
        "나는 왜 이것밖에 못하는 걸까.",  # 자기비하
        "업무량이 너무 많아서 숨이 막힌다.",  # 정서적 고갈
        "회사 정치에 지쳤다. 불만이 쌓인다.",  # 좌절/압박
    ]
    
    # 예측 (학습 전이므로 무작위 결과)
    labels, categories, probs = model.predict(sample_texts)
    
    print("\n예측 결과 (학습 전):")
    for text, cat, prob in zip(sample_texts, categories, probs):
        print(f"  '{text[:30]}...' → {cat}")
        print(f"    확률: {prob.cpu().numpy().round(3)}")
    
    # 3. 샘플 학습 데이터 생성
    print("\n" + "=" * 50)
    print("샘플 학습 데이터 생성 및 학습")
    print("=" * 50)
    
    # 샘플 학습 데이터 (실제로는 AI Hub 데이터 사용)
    train_texts = [
        # 정서적 고갈 (0)
        "오늘도 야근했다. 너무 지친다.", 
        "아무것도 하기 싫고 무력하다.",
        "매일 반복되는 일상이 지겹다.",
        "에너지가 바닥났다.",
        "쉬어도 피로가 안 풀린다.",
        
        # 좌절/압박 (1)
        "상사에게 또 깨졌다. 억울하다.",
        "노력해도 인정받지 못한다.",
        "불합리한 지시에 화가 난다.",
        "성과를 가로채여서 분하다.",
        "압박감에 숨이 막힌다.",
        
        # 부정적 대인관계 (2)
        "동료들이 나를 따돌리는 것 같다.",
        "팀 분위기가 너무 안 좋다.",
        "사람들과 대화하기 싫다.",
        "직장 내 갈등으로 힘들다.",
        "상사와의 관계가 최악이다.",
        
        # 자기비하 (3)
        "나는 능력이 없는 것 같다.",
        "실수할까봐 불안하다.",
        "다른 사람들보다 못한 것 같다.",
        "자신감이 바닥이다.",
        "이런 나 자신이 부끄럽다.",
    ]
    
    train_labels = [0]*5 + [1]*5 + [2]*5 + [3]*5
    
    train_dataset = BurnoutDataset(train_texts, train_labels)
    
    # 학습 (샘플이므로 짧게)
    model = train_model(
        model=model,
        train_dataset=train_dataset,
        epochs=5,
        batch_size=8,
        learning_rate=1e-3
    )
    
    # 4. 학습 후 예측
    print("\n" + "=" * 50)
    print("학습 후 예측 결과")
    print("=" * 50)
    
    labels, categories, probs = model.predict(sample_texts)
    
    for text, cat, prob in zip(sample_texts, categories, probs):
        print(f"  '{text[:30]}...' → {cat}")
    
    # 5. 모델 저장
    # model.save_model("burnout_classifier.pt")
    
    print("\n완료!")


if __name__ == "__main__":
    main()
