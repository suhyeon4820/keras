import numpy as np    
# 데이터
x1 = np.array([range(1, 101), range(101, 201), range(301, 401)])
x2 = np.array([range(1001, 1101), range(1101, 1201), range(1301, 1401)])
y1 = np.array([range(101, 201)])

print(x1.shape)  #(3, 100)
print(y1.shape)  #(1, 100)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

print(x1.shape)  #(100, 3) input
print(y1.shape)  #(100, 1) output

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.6, random_state=66, shuffle = False) # x, y -> (train(60%), test(40%))
x1_val, x1_test, x2_val, x2_test, y1_val, y1_test = train_test_split(x1_test, x2_test, y1_test, test_size=0.5, random_state=66, shuffle = False) # test(40%) -> (val(50%), test(50%))

# 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#함수형 모델 1
input1 = Input(shape = (3, ))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

#함수형 모델 2 node수나 layer depth 상관 없음
input2 = Input(shape = (3, ))
dense21 = Dense(7)(input2)
dense22 = Dense(4)(dense21)
output2 = Dense(5)(dense22) #output1/2가 달라도 merge 가능-> 아직 hidden layer(y)

from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])
middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = output)
model.summary()

# #model.add
# model = Sequential() # 연속적인 model layer
# model.add(Dense(5, input_shape = (3, ))) # input_shape : 2차원 
# model.add(Dense(2))
# model.add(Dense(3))
# model.add(Dense(1)) # outputlayer (node : 1개)
#model.summary()


# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size = 1, validation_split = 0, validation_data = ([x1_val, x2_val], y1_val)) 

# 평가예측
loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size = 1)
print('mse:', mse)

# # 예측
x1_pred = np.array([[200, 201, 202], [300, 301, 302], [400, 401, 402]])
x2_pred = np.array([[200, 201, 202], [300, 301, 302], [400, 401, 402]])

x1_pred = np.transpose(x1_pred)
x2_pred = np.transpose(x2_pred)

y_pred = model.predict([x1_test, x2_test], batch_size = 1)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y_pred):
    return np.sqrt(mean_squared_error(y1_test, y_pred))
print('RMSE : ', RMSE(y1_test, y_pred)) # 오차값이기 때문에 값이 적을수록 좋음

# RMSE2
print('print r_mse :', np.sqrt(mse))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y_pred)
print('R2 : ', r2_y_predict)  #R2는 최대값이 1로 1에 가까울수록 정확함