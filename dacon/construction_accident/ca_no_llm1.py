import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer

# Embedding Vector 추출에 활용할 모델(jhgan/ko-sbert-sts) 불러오기
model = SentenceTransformer("jhgan/ko-sbert-sts", use_auth_token=False)

train = pd.read_csv("c:/data/dacon/construction_accident/train.csv")
test = pd.read_csv("c:/data/dacon/construction_accident/test.csv")
sample = pd.read_csv("c:/data/dacon/construction_accident/sample_submission.csv")

grouped = train.groupby("인적사고")

res = {}
cosine_res = []
for name, group in tqdm(grouped):
    plan = group["재발방지대책 및 향후조치계획"]
    vectors = np.stack(plan.apply(model.encode).to_numpy())
    similarity = cosine_similarity(vectors, vectors)
    cosine_res += similarity[similarity.mean(axis=1).argmax()].tolist()
    res[name] = plan.iloc[similarity.mean(axis=1).argmax()]

arr = cosine_res

# 0.1 단위로 구간을 지정
bins = np.arange(0, 1.1, 0.1)  # 0.0 ~ 1.0을 0.1 간격으로 나눔

# 히스토그램 계산
hist, bin_edges = np.histogram(arr, bins=bins)

# 결과 출력
for i in range(len(hist)):
    print(f"Range {bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}: {hist[i]}개")

# print(res)

res_v = {}
for k, v in res.items():
    res_v[k] = model.encode(v)

# print(sample.head(3))
# print(sample.info())

for i in range(len(test)):
    accident = test.loc[i, "인적사고"]
    sample.loc[i, "재발방지대책 및 향후조치계획"] = res[accident]
    sample.iloc[i, 2:] = res_v[accident]

# print(sample.info())

sample.to_csv(
    "c:/data/dacon/construction_accident/output/nl_submission_01.csv",
    index=False,
    encoding="utf-8-sig",
)
print("ca_no_llm1.py Done.")
# https://dacon.io/competitions/official/236455/mysubmission
