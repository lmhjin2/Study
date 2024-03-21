import rasterio
import matplotlib.pyplot as plt
import numpy as np

# 밴드 조합 함수 정의
def get_combined_image(image_path, bands):
    # 이미지 데이터 로드
    with rasterio.open(image_path) as img:
        img_data = img.read(bands)  # 선택한 밴드 읽기
        
    # 밴드 데이터를 float32 타입으로 변환하고 정규화
    img_data = np.float32(img_data)
    img_data /= img_data.max()  # 각 밴드의 최대값으로 나누어 정규화
    
    # 밴드 순서 변경 (Rasterio는 CHW 형태로 로드하지만, Matplotlib는 HWC 형태를 요구)
    img_data = np.transpose(img_data, (1, 2, 0))
    
    return img_data

# 메인 시각화 함수
def visualize_image_and_mask(image_path, mask_path, bands, num):
    # 조합된 이미지 생성
    combined_image = get_combined_image(image_path.format(num=num), bands)
    
    # 마스크 데이터 로드
    with rasterio.open(mask_path.format(num=num)) as mask:
        mask_data = mask.read(1)  # 첫 번째 밴드만 읽기
        
    # 시각화
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # 조합된 이미지 시각화
    axs[0].imshow(combined_image)
    axs[0].set_title('Combined Bands Image')
    axs[0].axis('off')
    
    # 마스크 시각화
    axs[1].imshow(mask_data, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 사용 예시
num = 26  # 이미지 번호
image_path = 'c:/Study/aifactory/dataset/train_img/train_img_{num}.tif'
mask_path = 'c:/Study/aifactory/dataset/train_mask/train_mask_{num}.tif'
bands = (7,6,4)  # 밴드 조합 예시

visualize_image_and_mask(image_path, mask_path, bands, num)