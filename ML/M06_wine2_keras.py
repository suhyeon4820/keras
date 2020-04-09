import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv',
                   sep=";", encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)
print(x.shape) #(4989, 11)
print(y.shape) #(4898, )
print(x)
print(y)

label_encoder=LabelEncoder()
label_ids=label_encoder.fit_transform(y)

onehot_encoder=OneHotEncoder(sparse=False)
reshaped=label_ids.reshape(len(label_ids), 1)
onehot=onehot_encoder.fit_transform(reshaped)


x_train, x_test, y_train, y_test = train_test_split(x, onehot, test_size= 0.2, train_size = 0.8)

# 모델
model = Sequential()
model.add(Dense(64, input_shape= (11,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

#모델 컴파일
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

#모델 실행
model.fit(x_train, y_train, epochs=100, batch_size=10)

loss, accuracy = model.evaluate(x_train, y_train, batch_size = 1)
print('정확도:', accuracy)