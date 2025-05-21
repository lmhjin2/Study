import random
import numpy as np
import os

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import cv2

import zipfile
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CFG = {"EPOCHS": 10, "LEARNING_RATE": 3e-4, "BATCH_SIZE": 16, "SEED": 42}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG["SEED"])  # Seed 고정


class CustomDataset(Dataset):
    def __init__(self, damage_dir, origin_dir, transform=None):
        self.damage_dir = damage_dir
        self.origin_dir = origin_dir
        self.transform = transform
        self.damage_files = sorted(os.listdir(damage_dir))
        self.origin_files = sorted(os.listdir(origin_dir))

    def __len__(self):
        return len(self.damage_files)

    def __getitem__(self, idx):
        damage_img_name = self.damage_files[idx]
        origin_img_name = self.origin_files[idx]

        damage_img_path = os.path.join(self.damage_dir, damage_img_name)
        origin_img_path = os.path.join(self.origin_dir, origin_img_name)

        damage_img = Image.open(damage_img_path).convert("RGB")
        origin_img = Image.open(origin_img_path).convert("RGB")

        if self.transform:
            damage_img = self.transform(damage_img)
            origin_img = self.transform(origin_img)

        return {"A": damage_img, "B": origin_img}


# 경로 설정
origin_dir = "./train_gt"  # 원본 이미지 폴더 경로
damage_dir = "./train_input"  # 손상된 이미지 폴더 경로
test_dir = "./test_input"  # test 이미지 폴더 경로

# 데이터 전처리 설정
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# 데이터셋 및 DataLoader 생성
dataset = CustomDataset(
    damage_dir=damage_dir, origin_dir=origin_dir, transform=transform
)
dataloader = DataLoader(
    dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True, num_workers=1
)


# U-Net 기반의 Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()

        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(
                    in_feat, out_feat, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, normalize=False)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(1024, 512, dropout=0.5)
        self.up3 = up_block(1024, 512, dropout=0.5)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8


# PatchGAN 기반의 Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [
                nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)
            ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(in_channels * 2, 64, normalization=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# 가중치 초기화 함수
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def ssim_score(true, pred):
    # 전체 RGB 이미지를 사용해 SSIM 계산 (channel_axis=-1)
    ssim_value = ssim(true, pred, channel_axis=-1, data_range=pred.max() - pred.min())
    return ssim_value


def masked_ssim_score(true, pred, mask):
    # 손실 영역의 좌표에서만 RGB 채널별 픽셀 값 추출
    true_masked_pixels = true[mask > 0]
    pred_masked_pixels = pred[mask > 0]

    # 손실 영역 픽셀만으로 SSIM 계산 (채널축 사용)
    ssim_value = ssim(
        true_masked_pixels,
        pred_masked_pixels,
        channel_axis=-1,
        data_range=pred.max() - pred.min(),
    )
    return ssim_value


def histogram_similarity(true, pred):
    # BGR 이미지를 HSV로 변환
    true_hsv = cv2.cvtColor(true, cv2.COLOR_BGR2HSV)
    pred_hsv = cv2.cvtColor(pred, cv2.COLOR_BGR2HSV)

    # H 채널에서 히스토그램 계산 및 정규화
    hist_true = cv2.calcHist([true_hsv], [0], None, [180], [0, 180])
    hist_pred = cv2.calcHist([pred_hsv], [0], None, [180], [0, 180])
    hist_true = cv2.normalize(hist_true, hist_true).flatten()
    hist_pred = cv2.normalize(hist_pred, hist_pred).flatten()

    # 히스토그램 간 유사도 계산 (상관 계수 사용)
    similarity = cv2.compareHist(hist_true, hist_pred, cv2.HISTCMP_CORREL)
    return similarity


# 모델 저장을 위한 디렉토리 생성
model_save_dir = "./saved_models"
os.makedirs(model_save_dir, exist_ok=True)

best_loss = float("inf")
lambda_pixel = 100  # 픽셀 손실에 대한 가중치

# Generator와 Discriminator 초기화
generator = UNetGenerator()
generator = generator.to(device)
discriminator = PatchGANDiscriminator()
discriminator = discriminator.to(device)

generator.apply(weights_init_normal).to(device)
discriminator.apply(weights_init_normal).to(device)

# 손실 함수 및 옵티마이저 설정
criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()

optimizer_G = optim.Adam(generator.parameters(), lr=CFG["LEARNING_RATE"])
optimizer_D = optim.Adam(discriminator.parameters(), lr=CFG["LEARNING_RATE"])

# 학습
for epoch in range(1, CFG["EPOCHS"] + 1):
    for i, batch in enumerate(dataloader):
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)

        # Generator 훈련
        optimizer_G.zero_grad()
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        # Discriminator 훈련
        optimizer_D.zero_grad()
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # 진행 상황 출력
        print(
            f"[Epoch {epoch}/{CFG['EPOCHS']}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]"
        )

        # 현재 에포크에서의 손실이 best_loss보다 작으면 모델 저장
        if loss_G.item() < best_loss:
            best_loss = loss_G.item()
            torch.save(
                generator.state_dict(),
                os.path.join(model_save_dir, "best_generator.pth"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(model_save_dir, "best_discriminator.pth"),
            )
            print(
                f"Best model saved at epoch {epoch}, batch {i} with G loss: {loss_G.item()} and D loss: {loss_D.item()}"
            )

# 저장할 디렉토리 설정
submission_dir = "./submission"
os.makedirs(submission_dir, exist_ok=True)


# 이미지 로드 및 전처리
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원을 추가합니다.
    return image


# 모델 경로 설정
generator_path = os.path.join(model_save_dir, "best_generator.pth")

# 모델 로드 및 설정
model = UNetGenerator().to(device)
model.load_state_dict(torch.load(generator_path))
model.eval()

# 파일 리스트 불러오기
test_images = sorted(os.listdir(test_dir))

# 모든 테스트 이미지에 대해 추론 수행
for image_name in test_images:
    test_image_path = os.path.join(test_dir, image_name)

    # 손상된 테스트 이미지 로드 및 전처리
    test_image = load_image(test_image_path).to(device)

    with torch.no_grad():
        # 모델로 예측
        pred_image = model(test_image)
        pred_image = pred_image.cpu().squeeze(0)  # 배치 차원 제거
        pred_image = pred_image * 0.5 + 0.5  # 역정규화
        pred_image = pred_image.numpy().transpose(1, 2, 0)  # HWC로 변경
        pred_image = (pred_image * 255).astype("uint8")  # 0-255 범위로 변환

        # 예측된 이미지를 실제 이미지와 같은 512x512로 리사이즈
        pred_image_resized = cv2.resize(
            pred_image, (512, 512), interpolation=cv2.INTER_LINEAR
        )

    # 결과 이미지 저장
    output_path = os.path.join(submission_dir, image_name)
    cv2.imwrite(output_path, cv2.cvtColor(pred_image_resized, cv2.COLOR_RGB2BGR))

print(f"Saved all images")

# 저장된 결과 이미지를 ZIP 파일로 압축
zip_filename = "submission.zip"
with zipfile.ZipFile(zip_filename, "w") as submission_zip:
    for image_name in test_images:
        image_path = os.path.join(submission_dir, image_name)
        submission_zip.write(image_path, arcname=image_name)

print(f"All images saved in {zip_filename}")

# https://dacon.io/competitions/official/236420/mysubmission
