import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

# Hyperparameter Setting
CFG = {
    "IMG_SIZE": 224,
    "EPOCHS": 20,
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 16,
    "SEED": 41,
}

# Seed 고정
tf.random.set_seed(CFG["SEED"])

# Train & Validation Split
df = pd.read_csv("c:/Study/dacon/bird/train.csv")
train, val = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=CFG["SEED"]
)

# Label-Encoding
le = LabelEncoder()
train["label"] = le.fit_transform(train["label"])
val["label"] = le.transform(val["label"])

# Image Data Generator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)


# Data Loading
def get_data_generator(df, datagen, batch_size=32):
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory="c:/Study/dacon/bird/",
        x_col="img_path",
        y_col="label",
        target_size=(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        batch_size=batch_size,
        class_mode="raw",
    )
    return generator


train_generator = get_data_generator(train, train_datagen, CFG["BATCH_SIZE"])
val_generator = get_data_generator(val, val_datagen, CFG["BATCH_SIZE"])

from keras.applications import (
    EfficientNetB0,
    EfficientNetB2,
    EfficientNetV2L,
    InceptionResNetV2,
    InceptionV3,
)


# Model Define
def create_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=(CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3),
    )
    base_model.trainable = True

    inputs = keras.Input(shape=(CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CFG["LEARNING_RATE"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = create_model(len(le.classes_))
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
# Train
history = model.fit(
    train_generator, epochs=CFG["EPOCHS"], callbacks=[es], validation_data=val_generator
)

# Inference
test = pd.read_csv("c:/Study/dacon/bird/test.csv")
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory="c:/Study/dacon/bird/",
    x_col="img_path",
    batch_size=CFG["BATCH_SIZE"],
    shuffle=False,
    target_size=(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
    class_mode=None,
)

preds = model.predict(test_generator)
preds = np.argmax(preds, axis=1)
preds = le.inverse_transform(preds)

# Submission
submit = pd.read_csv("c:/Study/dacon/bird/sample_submission.csv")
submit["label"] = preds
submit.to_csv(f"c:/Study/dacon/bird/output/0412_2.csv", index=False)
