
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


#1. 데이터 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris.csv", encoding = 'utf-8',
                        names=['a','b','c','d','y'])

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

print(x.shape)

# 모델
model = Sequential()
model.add(Dense(12, input_shape= (4,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

#모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

#모델 실행
model.fit(x_train, y_train, epochs=100, batch_size=10)

y_pred = model.predict(x_test)
print("정답률 :", classification_report(y_test, y_pred.round())) 
