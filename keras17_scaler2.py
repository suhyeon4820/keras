# scaling
# 스케일링은 자료 집합에 적용되는 전처리 과정으로 
# 모든 자료에 선형 변환을 적용하여 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다.
# 스케일링은 자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지하고 
# 독립 변수의 공분산 행렬의 조건수(condition number)를 감소시켜 최적화 과정에서의 안정성 및 수렴 속도를 향상시킨다.
# Minmaxscaler = normalization(정규화)
# RobustScaler(x) : 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환
# MaxabsScaler(x) : 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환

from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
           [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12],
           [20000, 30000, 40000], [30000, 40000, 50000], [40000, 50000, 60000], [100, 200, 300]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50000, 60000, 70000, 400])

print(x.shape) #(14, 3)
print(y.shape) #(14,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

# StandardScaler(x) : 평균이 0과 표준편차가 1이 되도록 변환
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)
# x_pred = scaler.transform(x_pred)
print(x)

# MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)
print(x.shape) #(14, 3)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) #(14, 3, 1)

# Dense층 생성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3, 1)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 10개 test
from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(x, y, train_size = 0.35, random_state = 66, shuffle=False)
print(x_test.shape)    # (4, 3)
print(x_train.shape)   # (10, 3)
print(y_test.shape)    # (4,)
print(y_train.shape)   # (10,)

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용
model.fit(x_train, y_train, epochs=100, batch_size = 1)

# 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print('loss, mae :', loss, mse)

x_input = array([[250, 260, 270]])  #(1, 3)
print(x_input.shape)
#x_input = x_input.reshape(1, 3)

y_pred = model.predict(x_test)
print('y_pred', y_pred)
print(y_pred.shape)   #(4, 1)
print(y_test.shape)   #(4,)


from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_pred)
print('R2 : ', r2_y_predict)  #R2는 최대값이 1로 1에 가까울수록 정확함