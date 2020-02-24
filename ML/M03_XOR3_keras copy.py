from sklearn import datasets, preprocessing
from sklearn import model_selection
from keras import datasets, utils
from keras import models, layers, initializers, losses, optimizers, metrics
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score

# #1. 데이터
x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_train = np.array([0, 1, 1, 0])


# #2. 모델
# model = LinearSVC()
model = Sequential()
model.add(Dense(5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='RMSprop',
              metrics=['binary_crossentropy'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)
# print("acc = ", accuracy_score([0,1, 1, 0], y_predict))
# print("acc = ", accuracy_score(y_train, y_predict))