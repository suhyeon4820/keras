import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from keras import models, layers, initializers, losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

#1. 데이터
iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8', names=['a', 'b', 'c', 'd', 'y'])

# 붓꽃 데이터를 레이블과 입력 데이터로 분리


# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,"y"]
x = iris_data.loc[:,['a','b','c','d']]

label_encoder=LabelEncoder()
label_ids=label_encoder.fit_transform(y)

onehot_encoder=OneHotEncoder(sparse=False)
reshaped=label_ids.reshape(len(label_ids), 1)
onehot=onehot_encoder.fit_transform(reshaped)

x_train, x_test, y_train, y_test = train_test_split(
    x, onehot, test_size= 0.2, train_size = 0.7, shuffle = True
)

# y = iris_data.loc[:, 'y']
# y = np.array(y)
# print(y.shape) #(150, )

# x = iris_data.loc[:, ['a', 'b', 'c', 'd']]
# x = np.array(x)
# print(x.shape) #(150, 4)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.7, shuffle=True)
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train)
# print(y_train.shape)
# print(y_test.shape)

# 모델
model = Sequential() # 연속적인 model layer
model.add(Dense(128, input_shape = (4, ), activation='relu')) # input_shape : 2차원 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu')) 
model.add(Dense(3, activation='softmax'))
model.summary()

# 훈련/학습하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # mse, mae 사용
model.fit(x_train, y_train, epochs=150, batch_size = 1)

# 평가하기
loss, accuracy = model.evaluate(x_train, y_train, batch_size = 1)
print('정확도:', accuracy)

y_predict = model.predict(x_test, batch_size = 1)
print(y_predict)