# https://dacon.io/competitions/official/236214/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time as tm

path = "c:/_data/dacon/dechul/"
train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# train_csv['대출등급'] = train_csv['대출등급'].replace({"A":0, "B":1, "C":2,
#                                                "D":3,"E":4,"F":5})
# ["주택소유상태", "대출목적", "대출등급"]
# unknown 같은거 결측치 처리 해야함.

# df = pd.DataFrame(train_csv)
# print(df)   # (96294, 14)
le_work_period = LabelEncoder()
# print(train_csv)    # (96294, 14)
le_work_period.fit(train_csv['근로기간'])
train_csv['근로기간'] = le_work_period.transform(train_csv['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])
# df = df[df.근로기간 != 'Unknown'] # 근로기간에 Unknown들어간 행 삭제
                         # .dropna() 를 안써도됨 이미 포함됨
# print(train_csv['근로기간'])      # (96294, )  int32        
                                                  
le_grade = LabelEncoder()
le_grade.fit(train_csv['대출등급'])
train_csv['대출등급'] = le_grade.transform(train_csv['대출등급'])
# print(train_csv['대출등급'])     # (96294,) dtype: int32           



# encoder.fit(df['대출등급']) # label 인코더로 동일값은 동일 숫자를 가지게 바뀜
# df['대출등급'] = encoder.transform(df['대출등급'])
le_purpose = LabelEncoder()
test_csv.iloc[34486,7] = '이사'     # 결혼 -> 이사 로 임의로 바꿈
le_purpose.fit(train_csv['대출목적'])
train_csv['대출목적'] = le_purpose.transform(train_csv['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])
# print(train_csv['대출목적'])    # (96294,) dtype: int32 
# encoder.fit(df['대출목적']) # label 인코더로 동일값은 동일 숫자를 가지게 바뀜
# df['대출목적'] = encoder.transform(df['대출목적'])

le_own = LabelEncoder()
le_own.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le_own.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])
# print(train_csv['주택소유상태'])    # (96294,) dtype: int32 
# encoder.fit(df['주택소유상태']) # label 인코더로 동일값은 동일 숫자를 가지게 바뀜
# df['주택소유상태'] = encoder.transform(df['주택소유상태'])

# encoder.fit(df['근로기간']) # label 인코더로 동일값은 동일 숫자를 가지게 바뀜
# df['근로기간'] = encoder.transform(df['근로기간'])

# df['대출기간'] = df['대출기간'].str.extract(r'(\d+)').astype(int)
# df['대출기간'] = df['대출기간'].str.split().str[0].astype(int)

le_loan_period = LabelEncoder()
le_loan_period.fit(train_csv['대출기간'])
train_csv['대출기간'] = le_loan_period.transform(train_csv['대출기간'])
test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])
# print(train_csv['대출기간'])    # (96294,) dtype: int32 
# print(df['대출기간'])
# print(train_csv.dtypes)
# print(test_csv.dtypes)

# x = df.drop(['대출등급'], axis = 1)
# y = df['대출등급']
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

y = y.values.reshape(-1,1)
y_ohe = OneHotEncoder(sparse=False).fit_transform(y)
# y_ohe = enc.transform(y_ohe)
# y_ohe = y_ohe.astype(int)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_ohe, stratify=y, train_size = 0.85, random_state = 0 )

# print(x_train) # (81849, 13)
# print(y_train.shape) # (81849, 7)

# print(df)   # (90623, 14)

#2
model = Sequential()
model.add(Dense(64, input_shape = (13,)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(7, activation = 'softmax'))

#3
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', mode = 'auto',
                   patience = 500, verbose = 1,
                   restore_best_weights = True)
start_time = tm.time()
model.fit(x_train, y_train, epochs = 5000, batch_size = 100,
          validation_split = 0.18, verbose = 2, callbacks = [es])
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4
results = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis =1)
y_submit = np.argmax(y_submit, axis=1)
y_submit = le_grade.inverse_transform(y_submit)

submission_csv['대출등급'] = y_submit
submission_csv.to_csv(path + "submission_0113_3.csv", index=False)

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro')

print('accuracy_score :', acc)
print('?', results[1])
print('run time', run_time)
print('loss', results[0])
print('f1 score', f1)

