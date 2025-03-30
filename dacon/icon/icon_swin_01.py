import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

train_df = pd.read_csv("c:/data/dacon/icon/train.csv")
test_df = pd.read_csv("c:/data/dacon/icon/test.csv")

# 라벨 인코딩 (문자 -> 숫자)
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["label"])

# 데이터셋 분할
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# 이미지 변환 (Swin 모델은 224x224 크기를 요구)
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Swin Transformer의 입력 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 정규화
    ]
)


# 커스텀 데이터셋 생성
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = (
            row.drop(["ID", "label"]).values.astype(np.uint8).reshape(32, 32)
        )  # 32x32 이미지 복원
        label = row["label"]

        # 1채널 -> 3채널 변환
        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            image = self.transform(image)

        return image, label


# 데이터셋 & 데이터로더 생성
train_dataset = ImageDataset(train_data, transform=transform)
val_dataset = ImageDataset(val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class SwinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinClassifier, self).__init__()
        # pretrained=False로 설정하여 가중치 없이 아키텍처만 가져옴
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,  # 여기를 False로 변경
            num_classes=num_classes,
        )

        # 가중치 초기화 함수 추가
        self._initialize_weights()

    def _initialize_weights(self):
        # Swin Transformer의 각 레이어에 대한 가중치 초기화
        for name, module in self.swin.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        return self.swin(x)


# 모델 초기화
num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinClassifier(num_classes).to(device)
print(torch.cuda.is_available())

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
# Swin Transformer에 권장되는 학습 설정
optimizer = optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=0.05  # 더 작은 학습률 사용
)  # 가중치 감쇠 추가


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_losses = []
    val_losses = []  # 검증 손실 저장 리스트 추가
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # 학습 단계
        model.train()
        total_train_loss, total_correct = 0, 0

        for images, labels in train_loader:
            images, labels = (
                images.to(device),
                labels.to(device).long(),
            )  # labels를 long 타입으로 변환

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = total_correct / len(train_loader.dataset)

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # 검증 단계
        model.eval()
        total_val_loss = 0  # 검증 손실 누적값
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = (
                    images.to(device),
                    labels.to(device).long(),
                )  # labels를 long 타입으로 변환
                outputs = model(images)

                # 검증 손실 계산
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)  # 평균 검증 손실
        val_acc = val_correct / len(val_loader.dataset)

        val_losses.append(avg_val_loss)  # 검증 손실 저장
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )
        print(f"Train Accuracy = {train_acc:.4f}, Val Accuracy = {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer, epochs=100
)

# plt 그래프 그리기
"""
import matplotlib.pyplot as plt
# 그래프 그리기
plt.figure(figsize=(12, 4))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')    # 검증 손실 추가
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_test=False):
        self.dataframe = dataframe
        self.transform = transform
        self.is_test = is_test  # 테스트 여부를 추가

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = (
            row.drop(["ID"], errors="ignore").values.astype(np.uint8).reshape(32, 32)
        )  # 32x32 이미지 복원

        # 1채널 -> 3채널 변환
        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            image = self.transform(image)

        # 테스트 데이터는 label이 없으므로 None 반환
        if self.is_test:
            return image
        else:
            label = row["label"]
            return image, label


# 테스트 데이터셋 생성
test_dataset = ImageDataset(test_df, transform=transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 예측 수행
model.eval()
predictions = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        predictions.extend(preds)

# 예측된 값을 원래 라벨로 변환
pred_labels = label_encoder.inverse_transform(predictions)

# 결과 저장
output_df = pd.DataFrame({"ID": test_df["ID"], "label": pred_labels})
output_df.to_csv("c:/data/dacon/icon/output/swin_transformer_02.csv", index=False)
# https://dacon.io/competitions/official/236459/mysubmission
