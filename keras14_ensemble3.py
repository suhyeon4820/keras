import numpy as np    
# 데이터
x1 = np.array([range(1, 101), range(101, 201), range(301, 401)])
y1 = np.array([range(301, 401), range(401, 501), range(501, 601)])
y2 = np.array([range(1, 101), range(101, 201), range(301, 401)])
y3 = np.array([range(1, 101), range(101, 201), range(301, 401)])

print(x1.shape)  #(3, 100)
print(y1.shape)  #(1, 100)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

print(x1.shape)  #(100, 3) input
print(y1.shape)  #(100, 1) output

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size = 0.6, random_state = 66, shuffle=False)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, test_size = 0.5, random_state = 66, shuffle=False)

y2_train, y2_test, y3_train, y3_test = train_test_split(y2, y3, train_size = 0.6, random_state = 66, shuffle=False)
y2_val, y2_test, y3_val, y3_test = train_test_split(y2_test, y3_test, test_size = 0.5, random_state = 66, shuffle=False)

print(y3_train.shape) #(60,3)
print(y3_test.shape)  #(20,3)
print(y3_val.shape)   #(20,3)

# 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#함수형 모델 1
input1 = Input(shape = (3, ))
dense1 = Dense(5)(input1)
dense2 = Dense(3)(dense1)
output1 = Dense(4)(dense2)

output_1 = Dense(2)(output1)   # 1번째 모델
output_1 = Dense(3)(output_1)

output_2 = Dense(4)(output1)   # 2번째 모델
output_2 = Dense(4)(output_2)
output_2 = Dense(3)(output_2)

output_3 = Dense(5)(output1)   # 3번째 모델
output_3 = Dense(3)(output_3)

model = Model(inputs = input1, outputs = [output_1, output_2, output_3])
model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mse, mae 사용
model.fit(x1_train, [y1_train, y2_train, y3_train], epochs=100, batch_size = 1, validation_split = 0, validation_data = (x1_val, [y1_val, y2_val, y3_val])) 

# 평가예측(loss : 4개(총 loss포함), mse : 3개(metrix))
aaa = model.evaluate(x1_test, [y1_test, y2_test, y3_test], batch_size = 1)
print('aaa:', aaa)
# print(aaa) # evaluate가 반환하는 값 : 총 7개
# a_list = np.array(aaa)
# print(a_list.shape)  #(7,)

# 예측
x1_pred = np.array([[200, 201, 202], [300, 301, 302], [400, 401, 402]])
x1_pred = np.transpose(x1_pred)

pred_bbb = model.predict(x1_pred, batch_size = 1)
print(pred_bbb)  

print('==================================================================')
y1_pred = model.predict(x1_test, batch_size = 1) #(20, 3)*3 ==(3, 20, 3)

#RMSE : 3개의 rmse구해서 평균값 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y1_pred):
    return np.sqrt(mean_squared_error(y1_test, y1_pred))
rmse1 = RMSE(y1_pred[0], y1_test)
rmse2 = RMSE(y1_pred[1], y2_test)
rmse3 = RMSE(y1_pred[2], y3_test)
rmse = (rmse1 + rmse2 + rmse3)/3
print('RMSE : ', rmse)
# print('RMSE : ', RMSE(y1_test, y1_pred)) # 오차값이기 때문에 값이 적을수록 좋음

# # RMSE2
# print('print r_mse :', np.sqrt(mse))

from sklearn.metrics import r2_score
# r2_y1_predict = r2_score(y1_test, y1_pred)
# print('R2 : ', r2_y1_predict)  #R2는 최대값이 1로 1에 가까울수록 정확함
r2_y_predict1 = r2_score(y1_test, y1_pred[0])
r2_y_predict2 = r2_score(y2_test, y1_pred[1])
r2_y_predict3 = r2_score(y3_test, y1_pred[2])
r2_y_predict = (r2_y_predict1 + r2_y_predict2 + r2_y_predict3)/3
print('R2 Score :', r2_y_predict)
