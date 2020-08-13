from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web


#Fazendo a filtragem dos dados para predição:
df = web.DataReader('ITUB4.SA', data_source='yahoo', start='2019/10/29', end='2020/01/30')
df = df.filter(["Open", "Close", "High", "Low", "Volume", "Adj Close"])
data = df.values

print(df)

x_test = data[0:60]

for b in range(60):
    x_test[b][5] = (b%60)/60 + 1/60
print(x_test)

df = df.filter(["Open", "Close", "High", "Low"])
data = df.values
y_test = data[60:]

#Escalando os valores e ajustando as medidas dos dados:
scaler = MinMaxScaler(feature_range=(0, 1))

x_test = scaler.fit_transform(x_test)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))


#Carregando o modelo previamente feito:
my_model = load_model('ITUB4.SA.h5')

#Fazendo a predição da ação e mostrando na seguinte forma: [["Open", "Close", "High", "Low"]]
prediction = my_model.predict(x_test)

result = np.array([[.0, .0, .0, .0, .0, .0]])

result[0][0] = prediction[0][0]
result[0][1] = prediction[0][1]
result[0][2] = prediction[0][2]
result[0][3] = prediction[0][3]
result[0][4] = 0
result[0][5] = 0
result = scaler.inverse_transform(result)

prediction = np.array([[result[0][0], result[0][1], result[0][2], result[0][3]]])

print(prediction)
print(y_test)