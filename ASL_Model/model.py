import pandas as pd
#Como o formato agora mudou é necessário utilizar o formato csv da biblioteca pandas

import matplotlib.pyplot as plt

train_df = pd.read_csv("data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/sign_mnist_valid.csv")

#armazendando os valores corretos de validação
y_train = train_df['label']
y_valid = valid_df['label']

#deletando os valores da tabela original já que agor eles se encontram em variáveis 
del train_df['label']
del valid_df['label']

#obtendo apenas os valores da tabela

x_train = train_df.values
x_valid = valid_df.values

#normalizando os dados
x_train = x_train / 255
x_valid = x_valid / 255


#Codificando para binario
import tensorflow.keras as keras
#Cada número será representado por uma sequência de 24 dígitos
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

#construindo o modelo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#iniciando o modelo
model = Sequential()
#Adicionando as camadas de processamento
model.add(Dense(units = 512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = num_classes, activation='softmax'))
model.summary()
#compilando
model.compile(loss= 'categorical_crossentropy', metrics = ['accuracy'])

#Treinando o modelo
history = model.fit(x_train,y_train, epochs = 20, verbose = 1, validation_data = (x_valid,y_valid))

#DISCUSSÂO DE RESULTADO: A precisão aparenta estar boa, no entanto a 
# precisão de validação, ou seja, a precisão com relação a dados novos está 
# baixa, isso indica que o programa não está conseguindo 
# analisar corretamente novos dados e está apenas memorizando os
# de treino(Overfitting)
