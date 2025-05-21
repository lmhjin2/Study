import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from sklearn.preprocessing import MinMaxScaler
import os

config = {  # best
    "learning_rate": 1e-5,  # 2e-5
    "epoch": 100,  # 30
    "batch_size": 64,  # 64
    "hidden_size": 64,  # 64
    "num_layers": 4,  # 2
    "output_size": 3,  # 3
}

CFG = SimpleNamespace(**config)

품목_리스트 = [
    "건고추",
    "사과",
    "감자",
    "배",
    "깐마늘(국산)",
    "무",
    "상추",
    "배추",
    "양파",
    "대파",
]


def process_data(raw_file, 산지공판장_file, 전국도매_file, 품목명, scaler=None):
    raw_data = pd.read_csv(raw_file)
    산지공판장 = pd.read_csv(산지공판장_file)
    전국도매 = pd.read_csv(전국도매_file)

    # 타겟 및 메타데이터 필터 조건 정의
    conditions = {
        "감자": {
            "target": lambda df: (df["품종명"] == "감자 수미")
            & (df["거래단위"] == "20키로상자")
            & (df["등급"] == "상"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["감자"],
                "품종명": ["수미"],
                "등급명": ["상"],
            },
            "도매": {
                "시장명": ["*전국도매시장"],
                "품목명": ["감자"],
                "품종명": ["수미"],
            },
        },
        "건고추": {
            "target": lambda df: (df["품종명"] == "화건")
            & (df["거래단위"] == "30 kg")
            & (df["등급"] == "상품"),
            "공판장": None,
            "도매": None,
        },
        "깐마늘(국산)": {
            "target": lambda df: (df["거래단위"] == "20 kg") & (df["등급"] == "상품"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["마늘"],
                "품종명": ["깐마늘"],
                "등급명": ["상"],
            },
            "도매": {
                "시장명": ["*전국도매시장"],
                "품목명": ["마늘"],
                "품종명": ["깐마늘"],
            },
        },
        "대파": {
            "target": lambda df: (df["품종명"] == "대파(일반)")
            & (df["거래단위"] == "1키로단")
            & (df["등급"] == "상"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["대파"],
                "품종명": ["대파(일반)"],
                "등급명": ["상"],
            },
            "도매": {
                "시장명": ["*전국도매시장"],
                "품목명": ["대파"],
                "품종명": ["대파(일반)"],
            },
        },
        "무": {
            "target": lambda df: (df["거래단위"] == "20키로상자")
            & (df["등급"] == "상"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["무"],
                "품종명": ["기타무"],
                "등급명": ["상"],
            },
            "도매": {"시장명": ["*전국도매시장"], "품목명": ["무"], "품종명": ["무"]},
        },
        "배추": {
            "target": lambda df: (df["거래단위"] == "10키로망대")
            & (df["등급"] == "상"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["배추"],
                "품종명": ["쌈배추"],
                "등급명": ["상"],
            },
            "도매": {
                "시장명": ["*전국도매시장"],
                "품목명": ["배추"],
                "품종명": ["배추"],
            },
        },
        "사과": {
            "target": lambda df: (df["품종명"].isin(["홍로", "후지"]))
            & (df["거래단위"] == "10 개")
            & (df["등급"] == "상품"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["사과"],
                "품종명": ["후지"],
                "등급명": ["상"],
            },
            "도매": {
                "시장명": ["*전국도매시장"],
                "품목명": ["사과"],
                "품종명": ["후지"],
            },
        },
        "상추": {
            "target": lambda df: (df["품종명"] == "청")
            & (df["거래단위"] == "100 g")
            & (df["등급"] == "상품"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["상추"],
                "품종명": ["청상추"],
                "등급명": ["상"],
            },
            "도매": {
                "시장명": ["*전국도매시장"],
                "품목명": ["상추"],
                "품종명": ["청상추"],
            },
        },
        "양파": {
            "target": lambda df: (df["품종명"] == "양파")
            & (df["거래단위"] == "1키로")
            & (df["등급"] == "상"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["양파"],
                "품종명": ["기타양파"],
                "등급명": ["상"],
            },
            "도매": {
                "시장명": ["*전국도매시장"],
                "품목명": ["양파"],
                "품종명": ["양파(일반)"],
            },
        },
        "배": {
            "target": lambda df: (df["품종명"] == "신고")
            & (df["거래단위"] == "10 개")
            & (df["등급"] == "상품"),
            "공판장": {
                "공판장명": ["*전국농협공판장"],
                "품목명": ["배"],
                "품종명": ["신고"],
                "등급명": ["상"],
            },
            "도매": {"시장명": ["*전국도매시장"], "품목명": ["배"], "품종명": ["신고"]},
        },
    }

    # 타겟 데이터 필터링
    raw_품목 = raw_data[raw_data["품목명"] == 품목명]
    target_mask = conditions[품목명]["target"](raw_품목)
    filtered_data = raw_품목[target_mask]

    # 다른 품종에 대한 파생변수 생성
    other_data = raw_품목[~target_mask]
    unique_combinations = other_data[["품종명", "거래단위", "등급"]].drop_duplicates()
    for _, row in unique_combinations.iterrows():
        품종명, 거래단위, 등급 = row["품종명"], row["거래단위"], row["등급"]
        mask = (
            (other_data["품종명"] == 품종명)
            & (other_data["거래단위"] == 거래단위)
            & (other_data["등급"] == 등급)
        )
        temp_df = other_data[mask]
        for col in ["평년 평균가격(원)", "평균가격(원)"]:
            new_col_name = f"{품종명}_{거래단위}_{등급}_{col}"
            filtered_data = filtered_data.merge(
                temp_df[["시점", col]],
                on="시점",
                how="left",
                suffixes=("", f"_{new_col_name}"),
            )
            filtered_data.rename(
                columns={f"{col}_{new_col_name}": new_col_name}, inplace=True
            )

    # 공판장 데이터 처리
    if conditions[품목명]["공판장"]:
        filtered_공판장 = 산지공판장
        for key, value in conditions[품목명]["공판장"].items():
            filtered_공판장 = filtered_공판장[filtered_공판장[key].isin(value)]

        filtered_공판장 = filtered_공판장.add_prefix("공판장_").rename(
            columns={"공판장_시점": "시점"}
        )
        filtered_data = filtered_data.merge(filtered_공판장, on="시점", how="left")

    # 도매 데이터 처리
    if conditions[품목명]["도매"]:
        filtered_도매 = 전국도매
        for key, value in conditions[품목명]["도매"].items():
            filtered_도매 = filtered_도매[filtered_도매[key].isin(value)]

        filtered_도매 = filtered_도매.add_prefix("도매_").rename(
            columns={"도매_시점": "시점"}
        )
        filtered_data = filtered_data.merge(filtered_도매, on="시점", how="left")

    # 수치형 컬럼 처리
    numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns
    filtered_data = filtered_data[["시점"] + list(numeric_columns)]
    filtered_data[numeric_columns] = filtered_data[numeric_columns].fillna(0)

    # 정규화 적용
    if scaler is None:
        scaler = MinMaxScaler()
        filtered_data[numeric_columns] = scaler.fit_transform(
            filtered_data[numeric_columns]
        )
    else:
        filtered_data[numeric_columns] = scaler.transform(
            filtered_data[numeric_columns]
        )

    return filtered_data, scaler


class AgriculturePriceDataset(Dataset):
    def __init__(self, dataframe, window_size=9, prediction_length=3, is_test=False):
        self.data = dataframe
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.is_test = is_test

        self.price_column = "평균가격(원)"
        self.numeric_columns = self.data.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        self.sequences = []
        if not self.is_test:
            for i in range(
                len(self.data) - self.window_size - self.prediction_length + 1
            ):
                x = (
                    self.data[self.numeric_columns]
                    .iloc[i : i + self.window_size]
                    .values
                )
                y = (
                    self.data[self.price_column]
                    .iloc[
                        i
                        + self.window_size : i
                        + self.window_size
                        + self.prediction_length
                    ]
                    .values
                )
                self.sequences.append((x, y))
        else:
            self.sequences = [self.data[self.numeric_columns].values]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if not self.is_test:
            x, y = self.sequences[idx]
            return torch.FloatTensor(x), torch.FloatTensor(y)
        else:
            return torch.FloatTensor(self.sequences[idx])


class PricePredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PricePredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)


품목별_predictions = {}
품목별_scalers = {}

pbar_outer = tqdm(품목_리스트, desc="품목 처리 중", position=0)
for 품목명 in pbar_outer:
    pbar_outer.set_description(f"품목별 전처리 및 모델 학습 -> {품목명}")
    train_data, scaler = process_data(
        "c:/data/dacon/price/train/train.csv",
        "c:/data/dacon/price/train/meta/TRAIN_산지공판장_2018-2021.csv",
        "c:/data/dacon/price/train/meta/TRAIN_전국도매_2018-2021.csv",
        품목명,
    )
    품목별_scalers[품목명] = scaler
    dataset = AgriculturePriceDataset(train_data)

    # 데이터를 train과 validation으로 분할
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, CFG.batch_size, shuffle=False)

    input_size = len(dataset.numeric_columns)

    model = PricePredictionLSTM(
        input_size, CFG.hidden_size, CFG.num_layers, CFG.output_size
    )
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), CFG.learning_rate)

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)

    for epoch in range(CFG.epoch):
        train_loss = train_model(model, train_loader, criterion, optimizer, CFG.epoch)
        val_loss = evaluate_model(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                f"c:/data/dacon/price/models/best_model_{품목명}.pth",
            )

        print(
            f"Epoch {epoch+1}/{CFG.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    print(f"Best Validation Loss for {품목명}: {best_val_loss:.4f}")

    품목_predictions = []

    ### 추론
    pbar_inner = tqdm(range(25), desc="테스트 파일 추론 중", position=1, leave=False)
    for i in pbar_inner:
        test_file = f"c:/data/dacon/price/test/TEST_{i:02d}.csv"
        산지공판장_file = f"c:/data/dacon/price/test/meta/TEST_산지공판장_{i:02d}.csv"
        전국도매_file = f"c:/data/dacon/price/test/meta/TEST_전국도매_{i:02d}.csv"

        test_data, _ = process_data(
            test_file,
            산지공판장_file,
            전국도매_file,
            품목명,
            scaler=품목별_scalers[품목명],
        )
        test_dataset = AgriculturePriceDataset(test_data, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch)
                predictions.append(output.numpy())

        predictions_array = np.concatenate(predictions)

        # 예측값을 원래 스케일로 복원
        price_column_index = test_data.columns.get_loc(test_dataset.price_column)
        predictions_reshaped = predictions_array.reshape(-1, 1)

        # 가격 열에 대해서만 inverse_transform 적용
        price_scaler = MinMaxScaler()
        price_scaler.min_ = 품목별_scalers[품목명].min_[price_column_index]
        price_scaler.scale_ = 품목별_scalers[품목명].scale_[price_column_index]
        predictions_original_scale = price_scaler.inverse_transform(
            predictions_reshaped
        )
        # print(predictions_original_scale)

        if np.isnan(predictions_original_scale).any():
            pbar_inner.set_postfix({"상태": "NaN"})
        else:
            pbar_inner.set_postfix({"상태": "정상"})
            품목_predictions.extend(predictions_original_scale.flatten())

    품목별_predictions[품목명] = 품목_predictions
    pbar_outer.update(1)

sample_submission = pd.read_csv("c:/data/dacon/price/sample_submission.csv")

for 품목명, predictions in 품목별_predictions.items():
    sample_submission[품목명] = predictions

# 결과 저장
sample_submission.to_csv("c:/data/dacon/price/submission/price01.csv", index=False)
# https://dacon.io/competitions/official/236381/mysubmission
