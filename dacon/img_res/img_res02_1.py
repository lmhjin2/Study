import os
import cv2
import random
import numpy as np
import skimage
import umap
import hdbscan
import pandas as pd
import torch

from glob import glob
from collections import Counter
from tqdm.auto import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

BATCH_SIZE = 16
SEED = 42

# https://huggingface.co/geolocal/StreetCLIP
# StreetCLIP is a robust foundation model for open-domain image geolocalization and other geographic and climate-related tasks.

clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
clip_model.to("cuda")
clip_processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

image_paths = sorted(glob("./data/train_gt/*.png"))

image_features = []
for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    image_paths_batch = image_paths[i : i + BATCH_SIZE]
    images = [Image.open(image_path) for image_path in image_paths_batch]
    pixel_values = clip_processor.image_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ].to("cuda")
    with torch.no_grad():
        image_features_row = clip_model.get_image_features(pixel_values).cpu().numpy()
    image_features.append(image_features_row)

train_embeddings = np.vstack(image_features)
train_embeddings = normalize(train_embeddings, norm="l2")
np.save("./preproc/train_embeddings", train_embeddings)
# train_embeddings = np.load('./preproc/train_embeddings.npy')

clusterable_embedding = umap.UMAP(
    n_neighbors=5,
    min_dist=0.0,
    n_components=2,
    random_state=SEED,
).fit_transform(train_embeddings)

plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], s=0.1)

labels = hdbscan.HDBSCAN(
    min_cluster_size=5,
).fit_predict(clusterable_embedding)

plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=labels, s=0.1)

len(set(labels)), sum(labels == -1)
counter = Counter([label for label in labels if label != -1])
min(counter.values()), np.median(list(counter.values())), max(counter.values())

print("label is 0")

for image_path in np.array(image_paths)[labels == 0]:
    image = Image.open(image_path)
    plt.imshow(image)
    plt.show()
print("label is 1")
for image_path in np.array(image_paths)[labels == 1]:
    image = Image.open(image_path)
    plt.imshow(image)
    plt.show()

train_df = pd.DataFrame(columns=["image", "label"])
train_df["image"] = [os.path.basename(image_path) for image_path in image_paths]
train_df["label"] = labels
train_df.to_csv("./preproc/train_preproc.csv", index=False)

test_image_paths = sorted(glob("./data/test_input//*.png"))
test_df = pd.DataFrame(columns=["image"])
test_df["image"] = [os.path.basename(image_path) for image_path in test_image_paths]
test_df.to_csv("./preproc/test_preproc.csv", index=False)

# https://dacon.io/competitions/official/236420/mysubmission
