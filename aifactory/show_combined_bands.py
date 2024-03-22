import rasterio
import matplotlib.pyplot as plt
import numpy as np

# 밴드 조합 함수 정의
def get_combined_image1(image_path, bands1):
    # 이미지 데이터 로드
    with rasterio.open(image_path) as img:
        img_data = img.read(bands1)  # 선택한 밴드 읽기
        
    # 밴드 데이터를 float32 타입으로 변환하고 정규화
    img_data = np.float32(img_data)
    img_data /= img_data.max()  # 각 밴드의 최대값으로 나누어 정규화
    
    # 밴드 순서 변경 (Rasterio는 CHW 형태로 로드하지만, Matplotlib는 HWC 형태를 요구)
    img_data = np.transpose(img_data, (1, 2, 0))
    
    return img_data
# 2번 안만들어도 됐을듯?
def get_combined_image2(image_path, bands2):
    with rasterio.open(image_path) as img:
        img_data2 = img.read(bands2)
    
    img_data2 = np.float32(img_data2)
    img_data2 /= img_data2.max()
    
    img_data2 = np.transpose(img_data2, (1, 2, 0))
    
    return img_data2

# 메인 시각화 함수
def visualize_image_and_mask(image_path, mask_path, bands1, bands2, num):
    # 조합된 이미지 생성
    combined_image1 = get_combined_image1(image_path.format(num=num), bands1)
    combined_image2 = get_combined_image2(image_path.format(num=num), bands2)
    
    # 마스크 데이터 로드
    with rasterio.open(mask_path.format(num=num)) as mask:
        mask_data = mask.read(1)  # 첫 번째 밴드만 읽기
        
    # 시각화 1
    fig, axs = plt.subplots(1, 3, figsize=(18, 9))
    
    # 조합된 이미지 시각화
    axs[0].imshow(combined_image1)
    axs[0].set_title(f'{bands1} combined')
    axs[0].axis('off')
    
    # 마스크 시각화
    axs[1].imshow(mask_data, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')
    
    # 시각화 2
    axs[2].imshow(combined_image2)
    axs[2].set_title(f'{bands2} combined')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()


# 기본색 비교 1 / 포인트색 비교 12, 13, 16 / 탐지 어려움 15, 16 / 모양확인 19
num = 23

image_path = 'c:/Study/aifactory/dataset/train_img/train_img_{num}.tif'
mask_path = 'c:/Study/aifactory/dataset/train_mask/train_mask_{num}.tif'

bands1 = (10,6,4)
bands2 = (10,7,4)

visualize_image_and_mask(image_path, mask_path, bands1, bands2, num)


# 후보
# 7,4,3 
# 7,5,2 b
# 7,5,3 
# 7,5,4 
# 7,6,2 g <- 기본
# 7,6,4 b
# 7,6,5 g
# 8,7,6 b
# 10,5,4 
# 10,7,2 
# 10,7,4 
# 10,7,5 b
# 10,7,6