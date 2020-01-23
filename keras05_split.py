import numpy as np    
# 데이터
x = np.array(range(1, 102))
y = np.array(range(1, 101))
x_train = x[:60]
y_train = y[:60]
x_test = x[60:80]
y_test = y[60:80]
x_val = x[80:101]
y_val = y[80:101]

print(x_train)
print(x_test)
print(x_val)

# # 모델구성
# from keras.models import Sequential
# from keras.layers import Dense

# model = Sequential() # 연속적인 model layer
# model.add(Dense(5, input_shape = (1, ))) # input_shape : 2차원 
# model.add(Dense(2))
# model.add(Dense(3)) # layer 2 추가(node : 10개)
# model.add(Dense(1)) # outputlayer(node : 1개)

# model.summary()

# # 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
# model.fit(x_train, y_train, epochs=150, batch_size = 1, validation_split = 0, validation_data = (x_val, y_val)) 

# # 평가예측
# loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
# print('mse:', mse)

# # 예측
# x_pred = np.array([101, 102, 103])
# bbb = model.predict(x_pred, batch_size = 1)
# print(bbb)

# aaa = model.predict(x_test, batch_size = 1)
# print(aaa)


