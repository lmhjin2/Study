import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from types import SimpleNamespace
import os


config = {
    "learning_rate": 2e-5,
    "epoch": 30,
    "batch_size": 64,
    "hidden_size": 64,
    "num_layers": 2,
    "output_size": 3
}

CFG = SimpleNamespace(**config)

품목_리스트 = ['건고추', '사과', '감자', '배', '깐마늘(국산)', '무', '상추', '배추', '양파', '대파']


def process_data(raw_file, 산지공판장_file, 전국도매_file, 품목명, scaler=None):
    raw_data = pd.read_csv(raw_file)
    산지공판장 = pd.read_csv(산지공판장_file)
    전국도매 = pd.read_csv(전국도매_file)

    # 타겟 및 메타데이터 필터 조건 정의
    conditions = {
    '감자': {
        'target': lambda df: (df['품종명'] == '감자 수미') & (df['거래단위'] == '20키로상자') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['감자'], '품종명': ['수미'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['감자'], '품종명': ['수미']}
    },
    '건고추': {
        'target': lambda df: (df['품종명'] == '화건') & (df['거래단위'] == '30 kg') & (df['등급'] == '상품'),
        '공판장': None, 
        '도매': None  
    },
    '깐마늘(국산)': {
        'target': lambda df: (df['거래단위'] == '20 kg') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['마늘'], '품종명': ['깐마늘'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['마늘'], '품종명': ['깐마늘']}
    },
    '대파': {
        'target': lambda df: (df['품종명'] == '대파(일반)') & (df['거래단위'] == '1키로단') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['대파'], '품종명': ['대파(일반)'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['대파'], '품종명': ['대파(일반)']}
    },
    '무': {
        'target': lambda df: (df['거래단위'] == '20키로상자') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['무'], '품종명': ['기타무'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['무'], '품종명': ['무']}
    },
    '배추': {
        'target': lambda df: (df['거래단위'] == '10키로망대') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['배추'], '품종명': ['쌈배추'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['배추'], '품종명': ['배추']}
    },
    '사과': {
        'target': lambda df: (df['품종명'].isin(['홍로', '후지'])) & (df['거래단위'] == '10 개') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['사과'], '품종명': ['후지'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['사과'], '품종명': ['후지']}
    },
    '상추': {
        'target': lambda df: (df['품종명'] == '청') & (df['거래단위'] == '100 g') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['상추'], '품종명': ['청상추'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['상추'], '품종명': ['청상추']}
    },
    '양파': {
        'target': lambda df: (df['품종명'] == '양파') & (df['거래단위'] == '1키로') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['양파'], '품종명': ['기타양파'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['양파'], '품종명': ['양파(일반)']}
    },
    '배': {
        'target': lambda df: (df['품종명'] == '신고') & (df['거래단위'] == '10 개') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['배'], '품종명': ['신고'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['배'], '품종명': ['신고']}
    }
    }

    # 타겟 데이터 필터링
    raw_품목 = raw_data[raw_data['품목명'] == 품목명]
    target_mask = conditions[품목명]['target'](raw_품목)
    filtered_data = raw_품목[target_mask]

    # 다른 품종에 대한 파생변수 생성
    other_data = raw_품목[~target_mask]
    unique_combinations = other_data[['품종명', '거래단위', '등급']].drop_duplicates()
    for _, row in unique_combinations.iterrows():
        품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
        mask = (other_data['품종명'] == 품종명) & (other_data['거래단위'] == 거래단위) & (other_data['등급'] == 등급)
        temp_df = other_data[mask]
        for col in ['평년 평균가격(원)', '평균가격(원)']:
            new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
            filtered_data = filtered_data.merge(temp_df[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
            filtered_data.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


    # 공판장 데이터 처리
    if conditions[품목명]['공판장']:
        filtered_공판장 = 산지공판장
        for key, value in conditions[품목명]['공판장'].items():
            filtered_공판장 = filtered_공판장[filtered_공판장[key].isin(value)]
        
        filtered_공판장 = filtered_공판장.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
        filtered_data = filtered_data.merge(filtered_공판장, on='시점', how='left')

    # 도매 데이터 처리
    if conditions[품목명]['도매']:
        filtered_도매 = 전국도매
        for key, value in conditions[품목명]['도매'].items():
            filtered_도매 = filtered_도매[filtered_도매[key].isin(value)]
        
        filtered_도매 = filtered_도매.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
        filtered_data = filtered_data.merge(filtered_도매, on='시점', how='left')

    # 수치형 컬럼 처리
    numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns
    filtered_data = filtered_data[['시점'] + list(numeric_columns)]
    filtered_data[numeric_columns] = filtered_data[numeric_columns].fillna(0)

    # 정규화 적용
    if scaler is None:
        scaler = MinMaxScaler()
        filtered_data[numeric_columns] = scaler.fit_transform(filtered_data[numeric_columns])
    else:
        filtered_data[numeric_columns] = scaler.transform(filtered_data[numeric_columns])

    return filtered_data, scaler

class AgriculturePriceDataset(Dataset):
    def __init__(self, dataframe, window_size=9, prediction_length=3, is_test=False):
        self.data = dataframe
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.is_test = is_test
        
        self.price_column = '평균가격(원)' 
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.sequences = []
        if not self.is_test:
            for i in range(len(self.data) - self.window_size - self.prediction_length + 1):
                x = self.data[self.numeric_columns].iloc[i:i+self.window_size].values
                y = self.data[self.price_column].iloc[i+self.window_size:i+self.window_size+self.prediction_length].values
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

# Define Time2Vec layer
class Time2Vec(nn.Module):
    def __init__(self, input_dim):
        super(Time2Vec, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.periodic = nn.Linear(input_dim, input_dim-1)
    
    def forward(self, x):
        linear_out = self.linear(x)
        periodic_out = torch.sin(self.periodic(x))
        return torch.cat([linear_out, periodic_out], dim=-1)

# Define Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(attended + x)
        feedforward = self.ff(x)
        return self.norm2(feedforward + x)

# Define main model architecture
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size, dropout):
        super(TimeSeriesTransformer, self).__init__()
        self.time2vec = Time2Vec(input_size)
        self.embedding = nn.Linear(input_size, hidden_size)
        self.position_encoding = self.generate_position_encoding(hidden_size, 10)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def generate_position_encoding(self, hidden_size, max_len):
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        b, s, f = x.shape
        x = self.time2vec(x)
        x = self.embedding(x)
        x = x + self.position_encoding[:, :s, :].to(x.device)
        x = self.dropout(x)
        
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        x = x.mean(dim=1)
        return self.output_layer(x)

# Define custom dataset class
class AgriculturePriceDataset(Dataset):
    def __init__(self, dataframe, window_size=9, prediction_length=3, is_test=False):
        self.data = dataframe
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.is_test = is_test
        
        self.price_column = [col for col in self.data.columns if '평균가격(원)' in col and len(col.split('_')) == 1][0]
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.sequences = []
        if not self.is_test:
            for i in range(len(self.data) - self.window_size - self.prediction_length + 1):
                x = self.data[self.numeric_columns].iloc[i:i+self.window_size].values
                y = self.data[self.price_column].iloc[i+self.window_size:i+self.window_size+self.prediction_length].values
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

# Training function with mixed precision training
def train_model(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Main training loop
def train_and_predict(품목_리스트):
    품목별_predictions = {}
    품목별_scalers = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pbar_outer = tqdm(품목_리스트, desc="품목 처리 중", position=0)
    for 품목명 in pbar_outer:
        pbar_outer.set_description(f"품목별 전처리 및 모델 학습 -> {품목명}")
        
        # Data preprocessing
        train_data, scaler = process_all_items_with_full_conditions("c:/data/dacon/price/train/train.csv", 
                                         "c:/data/dacon/price/train/meta/TRAIN_산지공판장_2018-2021.csv", 
                                         "c:/data/dacon/price/train/meta/TRAIN_전국도매_2018-2021.csv", 
                                         품목명)
        품목별_scalers[품목명] = scaler
        
        # Dataset and DataLoader setup
        dataset = AgriculturePriceDataset(train_data)
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_data, CFG.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, CFG.batch_size, shuffle=False, num_workers=4)

        # Model setup
        input_size = len(dataset.numeric_columns)
        model = TimeSeriesTransformer(
            input_size=input_size,
            hidden_size=CFG.hidden_size,
            num_layers=CFG.num_layers,
            num_heads=CFG.num_heads,
            output_size=CFG.output_size,
            dropout=CFG.dropout
        ).to(device)

        # Training setup
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scaler = GradScaler()

        # Training loop
        best_val_loss = float('inf')
        os.makedirs('models', exist_ok=True)

        for epoch in range(CFG.epoch):
            train_loss = train_model(model, train_loader, criterion, optimizer, scheduler, scaler, device)
            val_loss = evaluate_model(model, val_loader, criterion, device)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'models/best_model_{품목명}.pth')
            
            print(f'Epoch {epoch+1}/{CFG.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
        # Inference
        품목_predictions = []
        pbar_inner = tqdm(range(25), desc="테스트 파일 추론 중", position=1, leave=False)
        
        for i in pbar_inner:
            test_file = f"./test/TEST_{i:02d}.csv"
            산지공판장_file = f"./test/meta/TEST_산지공판장_{i:02d}.csv"
            전국도매_file = f"./test/meta/TEST_전국도매_{i:02d}.csv"
            
            test_data, _ = process_data(test_file, 산지공판장_file, 전국도매_file, 품목명, scaler=품목별_scalers[품목명])
            test_dataset = AgriculturePriceDataset(test_data, is_test=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            model.eval()
            predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    predictions.append(output.cpu().numpy())
            
            predictions_array = np.concatenate(predictions)
            
            # Inverse transform predictions
            price_column_index = test_data.columns.get_loc(test_dataset.price_column)
            predictions_reshaped = predictions_array.reshape(-1, 1)
            
            price_scaler = MinMaxScaler()
            price_scaler.min_ = 품목별_scalers[품목명].min_[price_column_index]
            price_scaler.scale_ = 품목별_scalers[품목명].scale_[price_column_index]
            predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
            
            if np.isnan(predictions_original_scale).any():
                pbar_inner.set_postfix({"상태": "NaN"})
            else:
                pbar_inner.set_postfix({"상태": "정상"})
                품목_predictions.extend(predictions_original_scale.flatten())
                
        품목별_predictions[품목명] = 품목_predictions
        pbar_outer.update(1)
    
    return 품목별_predictions

# Run training and prediction
품목별_predictions = train_and_predict(품목_리스트)

# Prepare submission file
sample_submission = pd.read_csv('c:/data/dacon/price/sample_submission.csv')
for 품목명, predictions in 품목별_predictions.items():
    sample_submission[품목명] = predictions

# Save results
sample_submission.to_csv('c:/data/dacon/price/submission/price01_tf.csv', index=False)
# https://dacon.io/competitions/official/236381/mysubmission
