import pandas as pd
import numpy as np
import os
import random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFingerprintGenerator  # MorganGenerator를 사용하기 위해 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#1
CFG = {
    'NBITS': 2048,
    'SEED': 42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG['SEED'])  # Seed 고정

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS'])

# SMILES 데이터를 분자 지문으로 변환
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = morgan_generator.GetFingerprint(mol)  # MorganGenerator 사용
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))

chembl_data = pd.read_csv('c:/data/dacon/medicine/train.csv')
chembl_data.head()
# print(chembl_data.shape) # (1952, 15) 

train = chembl_data[['Smiles', 'pIC50']].copy()  # .copy()를 사용하여 경고 해결
train.loc[:, 'Fingerprint'] = train['Smiles'].apply(smiles_to_fingerprint)  # .loc를 사용하여 경고 해결

train_x = np.stack(train['Fingerprint'].values)
train_y = train['pIC50'].values

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

print(train_x.shape, train_y.shape)   # (1561, 2048) / (1561,)
print(val_x.shape, val_y.shape)       # (391, 2048)  / (391,)

#2
model = Sequential()
model.add(Dense(256, input_shape=(2048,)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 300 ,
                   restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience = 100,
                    verbose=1, mode='min')

hist = model.fit(train_x, train_y, epochs = 2000,
                 batch_size=352, callbacks=[es, rlr],
                 validation_split=0.12, verbose=1)

#4
def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

loss = model.evaluate(val_x, val_y)

# Validation 데이터로부터의 학습 모델 평가
val_y_pred = model.predict(val_x)
mse = mean_squared_error(pIC50_to_IC50(val_y), pIC50_to_IC50(val_y_pred))
rmse = np.sqrt(mse)

print(f'RMSE: {rmse}')

# test.csv 데이터로부터의 학습 모델 평가
test = pd.read_csv('c:/data/dacon/medicine/test.csv')
test.loc[:, 'Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)
test_x = np.stack(test['Fingerprint'].values)

test_y_pred = model.predict(test_x)
submit = pd.read_csv('c:/data/dacon/medicine/sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(test_y_pred)
submit.head()

submit.to_csv('c:/data/dacon/medicine/submission/03_CM02.csv', index=False)
# https://dacon.io/competitions/official/236336/mysubmission
# leader board : 0.1111111111
