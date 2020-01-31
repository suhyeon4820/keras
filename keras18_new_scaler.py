import numpy as np

x = np.array(range(1, 21))
y = np.array(range(1, 21))

x = x.reshape(20, 1)

print(x.shape) #(20, 1)
print(y.shape) #(20)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

print(x_train)
print(x_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(x_test)