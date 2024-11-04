import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 데이터 로드 함수
def load_data(data_dir):
    images = []
    labels = []
    class_mapping = {"0.양호": 0, "1.경증": 1, "2.중등도": 2, "3.중증": 3}
    conditions = ["1.미세각질", "2.피지과다", "3.모낭사이홍반", "4.모낭홍반농포", "5.비듬", "6.탈모"]

    for condition in conditions:
        image_path = os.path.join(data_dir, "원천데이터", condition)
        label_path = os.path.join(data_dir, "라벨링데이터", condition)

        for severity in class_mapping.keys():
            image_severity_path = os.path.join(image_path, severity)
            label_severity_path = os.path.join(label_path, severity)

            if not os.path.exists(image_severity_path):
                print(f"경로가 존재하지 않습니다: {image_severity_path}")
                continue

            for img_file in os.listdir(image_severity_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(image_severity_path, img_file)
                    img = cv2.imread(img_path)

                    if img is None:
                        print(f"Warning: 이미지 로드 실패 - {img_path}")
                        continue  # 이미지 로드 실패 시 건너뛰기

                    images.append(cv2.resize(img, (224, 224)) / 255.0)
                    
                    # JSON 라벨 파일 경로
                    label_file_path = os.path.join(label_severity_path, img_file.replace(".jpg", ".json"))
                    if os.path.exists(label_file_path):
                        with open(label_file_path, 'r') as f:
                            label_data = json.load(f)
                            labels.append(class_mapping[severity])

    images = np.array(images)
    labels = np.array(labels)
    
    if len(images) == 0:
        raise ValueError("이미지가 로드되지 않았습니다. 경로 또는 파일 형식을 확인하세요.")

    return images, labels

# 데이터 경로 설정
data_dir = 'c:/data/scalp'  # 경로 수정

# 데이터 로드
try:
    X, y = load_data(data_dir)
except ValueError as e:
    print(e)
    exit()

# 데이터셋 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# 모델 구성
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 전이 학습을 위해 가중치 고정

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 각 증상의 상태를 양호, 경증, 중등도, 중증으로 분류
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))

# 모델 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
