import numpy as np
import pandas as pd
from numpy import array

df1 = pd.read_csv('samsung.csv', index_col = 0, header = 0, encoding='cp949', sep = ',')
print(df1)
print(df1.shape) #(426, 5)

df2 = pd.read_csv('kospi200.csv', index_col = 0, header = 0, encoding='cp949', sep = ',')
print(df2)
print(df2.shape) #(426, 5)

# KOSPI200 거래량
for i in range(len(df2.index)):
    df2.iloc[i, 4] = int(df2.iloc[i, 4].replace(',', ''))
    
# 삼성전자 거래량
for i in range(len(df1.index)):
    for j in range(len(df1.iloc[i])):
        df1.iloc[i, j] = int(df1.iloc[i, j].replace(',', ''))
        
print(df1)
print(df2)

df1 = df1.sort_values(['일자'], ascending='True')
df2 = df2.sort_values(['일자'], ascending='True')
print(df1)

df1 = df1.values
df2 = df2.values

print(type(df1), type(df2))
print(df1.shape, df2.shape)

np.save('./samsung/data/samsung.npy', arr=df1)
np.save('./samsung/data/kospi200.npy', arr=df2)

