from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape)  #(10,)
print(in_seq2.shape)  #(10,)
print(out_seq.shape)  #(10,)

in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

print(in_seq1.shape)  #(10, 1)
print(in_seq2.shape)  #(10, 1)
print(out_seq.shape)  #(10, 1)

dataset = np.hstack((in_seq1, in_seq2, out_seq))
n_steps = 3

print(dataset) #(10, 3)

x, y = split_sequence(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])

print(x.shape)  #(8, 3, 2) = 48
print('=======================================')
print(x) 
print(y.shape)  #(8,)

# 1. 함수분석
# 2. DNN 모델
# 3. 지표는 loss
# 4. [[90, 95], [100, 105], [110, 115]] predict

x= x.reshape(8, 6) 

model = Sequential()
model.add(Dense(20, input_shape=(6,)))
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
print('loss:', loss)

# 평가예측
x_input = array([[90, 95], [100, 105], [110, 115]])  
x_input = x_input.reshape(1, 6)
print(x_input.shape)

y_pred = model.predict(x_input, batch_size = 1)
print(y_pred)

