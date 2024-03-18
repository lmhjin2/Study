# from transformers import TFSegformerForSemanticSegmentation

# model_checkpoint = "nvidia/mit-b0"
# id2label = {0: "outer", 1: "inner", 2: "border"}
# label2id = {label: id for id, label in id2label.items()}
# num_labels = len(id2label)
# model = TFSegformerForSemanticSegmentation.from_pretrained(
#     model_checkpoint,
#     num_labels=num_labels,
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True,
# )
# model. summary()

# model.compile(optimizer='adam')
###################################################################################################
######### tensor 2.11.0 이상
import keras_cv
import keras_core as keras
import numpy as np
# Using the class with a backbone:

import tensorflow as tf
import keras_cv

images = np.ones(shape=(1, 96, 96, 3))
labels = np.zeros(shape=(1, 96, 96, 1))
backbone = keras_cv.models.MiTBackbone.from_preset("segformer_b0")
model = keras_cv.models.segmentation.SegFormer(
    num_classes=1, backbone=backbone,
)
model.summary()
# # Evaluate model
# model(images)

# # Train model
# model.compile(
#     optimizer="adam",
#     loss=keras.losses.BinaryCrossentropy(from_logits=False),
#     metrics=["accuracy"],
# )
# model.fit(images, labels, epochs=3)
###################################################################################################
