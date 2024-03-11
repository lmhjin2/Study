import os
import warnings
warnings.filterwarnings("ignore")
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import threading
import random
import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import joblib

"""&nbsp;

## 사용할 함수 정의
"""
np.random.seed(0) 
random.seed(42)           
tf.random.set_seed(7)

from keras.applications import InceptionV3

inceptionv3 = InceptionV3(include_top=False, weights='mscoco')
# Load model directly
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

processor = AutoImageProcessor.from_pretrained("badmatr11x/semantic-image-segmentation")
model = SegformerForSemanticSegmentation.from_pretrained("badmatr11x/semantic-image-segmentation")
