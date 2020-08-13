import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU

stocks = ['ITSA4.SA', 'BBAS3.SA', 'BBDC4.SA', 'ITUB4.SA', 'PETR4.SA', 'GGBR4.SA', 'ABEV3.SA', 'CIEL3.SA', 'WEGE3.SA', 'BRFS3.SA', 'VALE3.SA', 'USIM5.SA', 'CSNA3.SA']
for stock in stocks:
#Tratamento de dados
    df = pd.read_json("Data/{}.json".format(stock))

    data = df.filter(['X_TRAIN'])

    x_train = data.values

    data = df.filter(['Y_TRAIN'])

    y_train = data.values

    x_train_data = []
    y_train_data = []

    scaler = MinMaxScaler(feature_range=(0,1))

    for i in range(x_train.shape[0]):
        x_train_data.append(scaler.fit_transform(x_train[i][0]))
        y_train_data.append(y_train[i][0])


    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    y_train_data = scaler.fit_transform(y_train_data)

    #Modelo de predição
    model = Sequential()
    model.add(LSTM(60, return_sequences= True, input_shape = (x_train_data.shape[1], x_train_data.shape[2])))
    model.add(SimpleRNN(240, return_sequences=True))
    model.add(LSTM(240, return_sequences= False))
    model.add(Dense(240, activation='softmax'))
    model.add(Dense(240, activation='relu'))
    model.add(Dense(480, activation='softmax'))
    model.add(Dense(600, activation='softmax'))
    model.add(Dense(480, activation='softmax'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(60, activation='softmax'))
    model.add(Dense(4))

    model.compile(optimizer='sgd', loss="mse", metrics=['acc', 'mae'])

    model.fit(x_train_data, y_train_data, batch_size=60, epochs=100, use_multiprocessing=True, validation_split=0.1, workers=4)

    #Salvamento dos pesos do modelo treinado
    model.save("{}.h5".format(stock))
