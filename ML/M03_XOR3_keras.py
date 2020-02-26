from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터
x_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_train = [0, 1, 1, 0]
x_train = np.array(x_train)
print(x_train.shape)

#2. 모델
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() # 연속적인 model layer
model.add(Dense(128, input_shape = (2, ), activation='relu')) # input_shape : 2차원 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # mse, mae 사용
model.fit(x_train, y_train, epochs=150, batch_size = 1)

#4. 평가예측
loss, accuracy = model.evaluate(x_train, y_train, batch_size = 1)
print('accuracy:', accuracy)

x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
x_test = np.array(x_test)
y_predict = model.predict(x_test, batch_size = 1)
print(y_predict)




