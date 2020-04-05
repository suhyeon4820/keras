from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#1. 데이터
iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8', names=['a', 'b', 'c', 'd', 'y'])

# 붓꽃 데이터를 레이블과 입력 데이터로 분리
y = iris_data.loc[:, 'y']
x = iris_data.loc[:, ['a', 'b', 'c', 'd']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.7, shuffle=True)

clf = SVC()
clf.fit(x_train, y_train)

# 학습하기
clf = SVC()
clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print('정확도:', accuracy_score(y_test, y_pred))