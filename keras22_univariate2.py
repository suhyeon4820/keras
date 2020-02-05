from numpy import array
import numpy as np

#(10, 3)
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):     # 10
        end_ix = i + n_steps           # 0 + 3 = 3   n_steps : 몇개씩 자르는지
        if end_ix > len(sequence)-1:  
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] 
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

dataset = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_steps = 3

x, y = split_sequence(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])

print(x.shape) # (7, 3)
print(y.shape) # (7,)

# 실습 DNN 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape = (3, )))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x, y, epochs=100, batch_size = 1, validation_split = 0) 

# 평가예측
loss, mse = model.evaluate(x, y, batch_size = 1)
print('loss, mae :', loss, mse)

# 평가예측
x_input = array([[90, 100, 110]])  

y_pred = model.predict(x_input, batch_size = 1)
print(y_pred)

