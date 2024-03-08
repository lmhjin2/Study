import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 데이터 불러오기
x, y = load_breast_cancer(return_X_y=True)

# 다항 특성 추가
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, random_state=777, train_size=0.8, stratify=y)

# 로지스틱 회귀 모델 초기화 및 학습
model = LogisticRegression()
model.fit(x_train, y_train)

# 예측 및 평가
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# 다항식 회귀 그래프 그리기
plt.figure(figsize=(10, 6))

# 원본 데이터와 예측값 비교를 위해 테스트 데이터 예측
y_pred = model.predict(x_test)

# 첫 번째 특성만 사용하여 그래프 생성
plt.scatter(x_test[:, 0], y_test, color='blue', label='Actual Data')
plt.scatter(x_test[:, 0], y_pred, color='red', marker='x', label='Predicted Data')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Comparison of Actual and Predicted Data')
plt.legend()
plt.grid(True)
plt.show()

## from GPT
# 이 코드는 다음을 수행합니다:

# 유방암 데이터를 로드합니다.
# 다항 특성을 추가합니다.
# 데이터를 학습 및 테스트 세트로 분할합니다.
# 로지스틱 회귀 모델을 초기화하고 학습합니다.
# 학습된 모델을 사용하여 테스트 데이터에 대한 예측을 수행하고 정확도를 출력합니다.
# 첫 번째 특성만을 사용하여 원본 데이터와 예측값을 시각화합니다.
# 위의 코드는 주어진 데이터를 기반으로 다항 특성을 추가한 후 로지스틱 회귀 모델을 학습하고, 
# 예측값과 실제값을 비교하는 그래프를 그리는 코드입니다. 
# 원본 데이터와 예측값을 시각화하여 모델의 성능을 평가할 수 있습니다.