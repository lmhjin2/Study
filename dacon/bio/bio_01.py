import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb

#1
train = pd.read_csv("c:/data/dacon/bio/train.csv")
test = pd.read_csv("c:/data/dacon/bio/test.csv")

le_subclass = LabelEncoder()
train['SUBCLASS'] = le_subclass.fit_transform(train['SUBCLASS'])

for i, label in enumerate(le_subclass.classes_):
    print(f"기존 레이블 : {label}, 변환 후 : {i}")

X = train.drop(columns=['SUBCLASS', 'ID'])
y_subclass = train['SUBCLASS']

categorical_columns = X.select_dtypes(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss' 
)

model.fit(X_encoded, y_subclass, verbose=1)

test_X = test.drop(columns=['ID'])
X_encoded = test_X.copy()
X_encoded[categorical_columns] = ordinal_encoder.transform(test_X[categorical_columns])
predictions = model.predict(X_encoded)
original_labels = le_subclass.inverse_transform(predictions)

submission = pd.read_csv("c:/data/dacon/bio/sample_submission.csv")
submission["SUBCLASS"] = original_labels
submission.to_csv('c:/data/dacon/bio/bio_01.csv', encoding='UTF-8-sig', index=False)
