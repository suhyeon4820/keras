import numpy as np    
# 데이터
x = np.array(range(1, 101))
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=66, shuffle = False) # x, y -> (train(60%), test(40%))
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=66, shuffle = False) # test(40%) -> (val(50%), test(50%))

# print(x_train)
# print(x_test)
# print(x_val)

# print(x_train)
# print(x_test)
# print(x_val)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() # 연속적인 model layer
model.add(Dense(5, input_shape = (1, ))) # input_shape : 2차원 
model.add(Dense(2))
model.add(Dense(3)) # layer 2 추가(node : 10개)
model.add(Dense(1)) # outputlayer (node : 1개)

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x_train, y_train, epochs=150, batch_size = 1, validation_split = 0, validation_data = (x_val, y_val)) 

# 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print('mse:', mse)

# # 예측
# x_pred = np.array([101, 102, 103])
# bbb = model.predict(x_pred, batch_size = 1)
# print(bbb)

aaa = model.predict(x_test, batch_size = 1)
print(aaa)

y_pred = model.predict(x_test, batch_size = 1)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE : ', RMSE(y_test, y_pred)) # 오차값이기 때문에 값이 적을수록 좋음

# RMSE2
print('print r_mse :', np.sqrt(mse))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_pred)
print('R2 : ', r2_y_predict)  #R2는 최대값이 1로 1에 가까울수록 정확함