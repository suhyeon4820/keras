# 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() # 연속적인 model layer
model.add(Dense(1, input_shape = (3, ))) # input_shape : 2차원 
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10)) # layer 2 추가(node : 10개)
model.add(Dense(1)) # outputlayer (node : 1개)

model.summary()

model.save('./save/savetest01.h5')
print('저장 완료')