from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model_P = Perceptron()
model_L = LogisticRegression()
model_K = KNeighborsClassifier()
model_D = DecisionTreeClassifier()
model_R = RandomForestClassifier()

models = [model_P, model_L,  model_K, model_D, model_R]

for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, "acc", results)

## 위에 분류 아래 회귀

model_P = Perceptron()
model_L = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()

for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, "acc", results)