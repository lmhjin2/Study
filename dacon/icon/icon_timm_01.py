import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToPILImage, ToTensor
from tqdm import tqdm

N_EPOCHS = 5
BATCH_SIZE = 8
LR = 5e-5
N_FOLDS = 10
SEED = 42

train = pd.read_csv("c:/data/dacon/icon/train.csv")
test = pd.read_csv("c:/data/dacon/icon/test.csv")

encoder = LabelEncoder()
train["label"] = encoder.fit_transform(train["label"])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for train_idx, valid_idx in skf.split(train.iloc[:, 2:], train["label"]):
    break


class CustomDataset(Dataset):
    def __init__(self, pixel_df, label_df=None, transform=None):
        self.pixel_df = pixel_df.reset_index(drop=True)
        self.label_df = (
            label_df.reset_index(drop=True) if label_df is not None else None
        )
        self.transform = transform

    def __len__(self):
        return len(self.pixel_df)

    def __getitem__(self, idx):
        # Reshape to (32, 32) from flattened data
        image = self.pixel_df.iloc[idx].values.astype(np.uint8).reshape(32, 32)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(
            0
        )  # shape: (1, 32, 32)

        if self.transform:
            image = self.transform(image)

        if self.label_df is not None:
            label = torch.tensor(self.label_df.iloc[idx], dtype=torch.long)
            return image, label
        else:
            return image


train_transform = Compose(
    [
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5]),
    ]
)

train_dataset = CustomDataset(
    pixel_df=train.iloc[train_idx, 2:],
    label_df=train.iloc[train_idx, 1],
    transform=train_transform,
)
valid_dataset = CustomDataset(
    pixel_df=train.iloc[valid_idx, 2:],
    label_df=train.iloc[valid_idx, 1],
    transform=train_transform,
)
test_dataset = CustomDataset(pixel_df=test.iloc[:, 1:], transform=train_transform)

loader_params = {"batch_size": BATCH_SIZE, "num_workers": 8, "pin_memory": True}

train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_params)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

model = timm.create_model(
    model_name="tf_efficientnet_b0.ns_jft_in1k",
    pretrained=False,
    num_classes=10,
    in_chans=1,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


best_loss = float("inf")
best_model = None

for epoch in range(N_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{N_EPOCHS}]")

    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Validate
    val_loss, val_acc = validate_one_epoch(model, valid_loader, criterion, device)

    print(
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%"
    )

    # Check for best model
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model

    scheduler.step()

best_model.eval()
preds = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference", leave=False):
        images = images.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs.data, 1)
        preds.extend(predicted.cpu().numpy())

# Decode predictions
pred_labels = encoder.inverse_transform(preds)

submission = pd.read_csv("c:/data/dacon/icon/sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("c:/data/dacon/icon/output/t_submission_01.csv", index=False)
# https://dacon.io/competitions/official/236459/mysubmission
