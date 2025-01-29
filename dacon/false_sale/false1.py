import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('./train.csv')

train.head()
train.info()

x = train.drop(['ID','허위매물여부'],axis=1)
y = train['허위매물여부']

# SimpleImputer : 평균 대체
mean_imputer = SimpleImputer(strategy='mean')

# 결측값을 평균으로 대체할 열 목록
columns_fill_mean = ['해당층', '총층','전용면적','방수', '욕실수','총주차대수']

# 학습 데이터에 fit 및 transform
x[columns_fill_mean] = mean_imputer.fit_transform(x[columns_fill_mean])

# Label Encoding 적용 열
label_encode_cols = ['중개사무소','게재일','제공플랫폼','방향']

label_encoders = {}
for col in label_encode_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col].astype(str))
    label_encoders[col] = le
    
# One-Hot Encoding 적용 열
one_hot_cols = ['매물확인방식', '주차가능여부']

# One-Hot Encoding 적용
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Train 데이터 변환
x_encoded = one_hot_encoder.fit_transform(x[one_hot_cols])
x_encoded_df = pd.DataFrame(x_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols), index=x.index)

# 기존 데이터와 병합
x = pd.concat([x.drop(columns=one_hot_cols), x_encoded_df], axis=1)

model = RandomForestClassifier(n_estimators=100,
                               criterion='gini',
                               max_depth=None,
                               random_state=42)
model.fit(x, y)

# Test 데이터 로드
test = pd.read_csv('./test.csv')

# Test 결측값 대체
test[columns_fill_mean] = mean_imputer.transform(test[columns_fill_mean])

# Label Encoding 
for col in label_encode_cols:
    if col in test.columns:
        le = label_encoders[col]
        test[col] = test[col].astype(str)
        unseen = set(test[col].unique()) - set(le.classes_)
        if unseen:
            le.classes_ = np.append(le.classes_, list(unseen))
        test[col] = le.transform(test[col])
        
# One-Hot Encoding
test_encoded = one_hot_encoder.transform(test[one_hot_cols])
test_encoded_df = pd.DataFrame(test_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols), index=test.index)

test = pd.concat([test.drop(columns=one_hot_cols), test_encoded_df], axis=1)

test.drop(columns=['ID'],inplace=True)
pred = pd.Series(model.predict(test))

submit = pd.read_csv('./sample_submission.csv')
submit['허위매물여부'] = pred
print(submit.head())
submit.to_csv('./baseline_submission.csv',index=False)
# https://dacon.io/competitions/official/236439/mysubmission

