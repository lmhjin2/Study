import rasterio
import matplotlib.pyplot as plt

# 위성 이미지 및 마스크 데이터 로드
num = 11
image_path = f'c:/Study/aifactory/dataset/train_img/train_img_{num}.tif'
mask_path = f'c:/Study/aifactory/dataset/train_mask/train_mask_{num}.tif'

# 위성 이미지 데이터 로드
with rasterio.open(image_path) as img:
    img_data = img.read()  # 모든 밴드 읽기

# 마스크 데이터 로드
with rasterio.open(mask_path) as mask:
    mask_data = mask.read(1)  # 첫 번째 밴드만 읽기 (마스크 데이터는 단일 밴드일 가능성이 높음)

# 시각화
fig, axs = plt.subplots(3, 4, figsize=(20, 15))  # 10개 밴드 + 1개 마스크 = 총 11개 시각화 필요

# 각 밴드별로 시각화
for i in range(10):
    ax = axs[i // 4, i % 4]
    ax.imshow(img_data[i], cmap='gray')
    ax.set_title(f'Band {i+1}')
    ax.axis('off')

# 마스크 시각화 (11번째 위치에 마스크 시각화)
axs[2, 2].imshow(mask_data, cmap='gray')
axs[2, 2].set_title('Mask')
axs[2, 2].axis('off')

# 나머지 subplot 비활성화
for ax in axs[2, 3:]:
    ax.axis('off')
plt.tight_layout()
plt.show()
