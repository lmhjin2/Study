import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boton"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 현재 사이킷런버전 1.3.0 보스턴 안됨. 그래서 삭제할거임
# pip uninstall scikit-learn
# pip uninstall scikit-image
# pip uninstall scikit-learn-intelex

# pip install scikit-learn==0.23.2