from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

boston = load_boston()

# 1. 데이터
x = boston.data
y = boston.target

print(x.shape) #(506, 13)
print(y.shape) #(50, )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, train_size = 0.7, shuffle = True)

# 2. 모델
from sklearn.linear_model import LinearRegression, Ridge, Lasso
model_l = LinearRegression()
model_r = Ridge()
model_la = Lasso()

# 3. 훈련
model_l.fit(x_train, y_train)
model_r.fit(x_train, y_train)
model_la.fit(x_train, y_train)

# 4. 평가
model1 = model_l.score(x_test, y_test)
print('LinearRegression:', model1)
model2 = model_r.score(x_test, y_test)
print('Ridge :', model2)
model3 = model_la.score(x_test, y_test)
print('Lasso:', model3)


