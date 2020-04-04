from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

#1. 데이터
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]
print(X.shape) #(768, 8)
print(Y.shape) #(768, )

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.6, random_state=66, shuffle = False)

print(X_train.shape) #(460, 8)
print(Y_train.shape) #(460)
print(X_test.shape)  #(308, 8)
print(Y_test.shape)  #(308, )

# 모델의 설정
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, Y_train)

y_predict = model.predict(X_test)
print(y_predict.shape)  #(308, )
print('acc=', accuracy_score(Y_test, y_predict))
