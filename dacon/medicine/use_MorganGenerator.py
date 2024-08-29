import pandas as pd
import numpy as np
import os
import random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFingerprintGenerator  # MorganGenerator를 사용하기 위해 추가
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

CFG = {
    'NBITS': 2048,
    'SEED': 42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG['SEED'])  # Seed 고정

# MorganGenerator 생성
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS'])

# SMILES 데이터를 분자 지문으로 변환
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = morgan_generator.GetFingerprint(mol)  # MorganGenerator 사용
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))

# 학습 ChEMBL 데이터 로드
chembl_data = pd.read_csv('c:/data/dacon/medicine/train.csv')
chembl_data.head()

# print(chembl_data.shape) # (1952, 15) 

train = chembl_data[['Smiles', 'pIC50']].copy()  # .copy()를 사용하여 경고 해결
train.loc[:, 'Fingerprint'] = train['Smiles'].apply(smiles_to_fingerprint)  # .loc를 사용하여 경고 해결

train_x = np.stack(train['Fingerprint'].values)
train_y = train['pIC50'].values

# 학습 및 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestRegressor(random_state=CFG['SEED'])
model.fit(train_x, train_y)

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

# Validation 데이터로부터의 학습 모델 평가
val_y_pred = model.predict(val_x)
mse = mean_squared_error(pIC50_to_IC50(val_y), pIC50_to_IC50(val_y_pred))
rmse = np.sqrt(mse)

print(f'RMSE: {rmse}')

test = pd.read_csv('c:/data/dacon/medicine/test.csv')
test.loc[:, 'Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

test_x = np.stack(test['Fingerprint'].values)

test_y_pred = model.predict(test_x)

submit = pd.read_csv('c:/data/dacon/medicine/sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(test_y_pred)
submit.head()

submit.to_csv('c:/data/dacon/medicine/submission/use_MG.csv', index=False)
