#1. 데이터
import numpy as np
x = np.array([range(1,101),range(101,201),range(301,401)])
y = np.array([range(101,201)])

# print(x.shape) # (3,100)
# print(y.shape) # (1,100)

x = np.transpose(x) # 행과 열을 바꿔줌
y = np.transpose(y)

print(x.shape) 
print(y.shape) 

from sklearn.model_selection import train_test_split

# train, validation , test  6:2:2 분할
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.4, shuffle=False)
x_val1, x_test, y_val2, y_test = train_test_split(x_val,y_val, test_size = 0.5, shuffle=False)

#2. 모델 구성
from keras.models import load_model, Model
from keras.layers import Dense
model = load_model('./save/savetest01.h5')
# x = model.output
# x = Dense(5, name = 'a')(x)
# x = Dense(4, name = 'b')(x)
# x = Dense(1, name = 'c')(x)
# model = Model(inputs = model.input,outputs = x)
model.add(Dense(5,name = 'a'))
model.add(Dense(1,name = 'b'))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])

from keras.callbacks import EarlyStopping # Epoch 을 많이 돌린 후, 특정 시점에서 멈추는 것, 과적합 방지
from keras.callbacks import TensorBoard

tb_hist = TensorBoard(log_dir= './graph', 
                      histogram_freq=0, # 통상적으로 써줌
                      write_graph=True, 
                      write_images=True)  # cmd에 log_dir폴더 전까지만가서 'tensorboard --logdir=./폴더명' 실행, 후에 크롬에 'localhost:6006' 실행

early_stopping = EarlyStopping(monitor='loss', patience= 20, mode = 'auto')

model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val,y_val), callbacks = [early_stopping, tb_hist])

#4. 평가
loss, mse = model.evaluate(x_test, y_test,batch_size=1)
print('mse : ', mse)

x_prd = np.array([[201,202,203],[204,205,206],[207,208,209]])
x_prd = np.transpose(x_prd)
aaa = model.predict(x_prd,batch_size=1)
print(aaa)


y_predict = model.predict(x_test,batch_size=1)
# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

# R2 구하기 0<r2<1 결정계수
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("r2:",r2_y_predict)