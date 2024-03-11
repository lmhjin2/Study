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

# inceptionv3 = InceptionV3(include_top=False, weights='imagenet')

# Load model directly
# from transformers import SegformerFeatureExtractor, TFSegformerForSemanticSegmentation
# from PIL import Image
# feature_extractor = SegformerFeatureExtractor()
# model = TFSegformerForSemanticSegmentation()

# model.summary()
###############################################################################################################

# from transformers import TFSegformerForSemanticSegmentation, SegformerConfig

# # 모델 구성 설정
# config = SegformerConfig.from_pretrained('google/segformer-b0')

# # 모델 초기화
# model = TFSegformerForSemanticSegmentation(config)

# model.summary()

###############################################################################################################



'''###############################################################################################################
from keras_cv_attention_models import swin_transformer_v2

# Will download and load pretrained imagenet weights.
mm = swin_transformer_v2.SwinTransformerV2Tiny_window8(pretrained="imagenet")

# Run prediction
import tensorflow as tf
from tensorflow import keras
from skimage.data import chelsea
imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
# [('n02124075', 'Egyptian_cat', 0.77100605), ('n02123159', 'tiger_cat', 0.04094378), ...]
'''

###############################################################################################################

from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import torch

# load MaskFormer fine-tuned on COCO panoptic segmentation
processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for postprocessing
result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
predicted_panoptic_map = result["segmentation"]
