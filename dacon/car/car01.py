import pandas as pd

train = pd.read_csv("c:/data/dacon/car/train.csv")
test = pd.read_csv("c:/data/dacon/car/test.csv")
submission = pd.read_csv("c:/data/dacon/car/sample_submission.csv")

user = str()
for i in range(40):
    user += (
        "ID: "
        + str(test["ID"][i])
        + " title: "
        + str(test["title"][i])
        + " notes: "
        + str(test["notes"][i])
        + "/n"
    )

submission.at[0, "system"] = (
    "당신은 자동차 데이터 전문가입니다. 입력으로 들어오는 데이터가 자동차 관련 데이터라면 1, 아니라면 0으로 판별해주세요. 답변해야할 데이터는 ID : TEST_00 ~ TEST_39로 40개며, 각 데이터 당 행으로 구분하여 0 또는 1로 답변해주세요. 40행 안 나오면 전원 차단합니다"
)
submission.at[0, "user"] = user

submission.to_csv("c:/data/dacon/car/submission/car01.csv", index=False)
