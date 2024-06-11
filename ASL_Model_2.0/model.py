#CONVOLUÇÃO ocorre quando se aplica uma função em outra função, no caso de imagens
#se utiliza esse método multiplicando os valores do kernel e o colocadno em uma nova 
#matriz, por exemplo: uma imagem 16x16 é reduzida em 4 matrizes (kernels) 4x4, tem seus valores 
#multiplicados e é adicionada a uma nova matriz 2x2, em seguida utiliza-se o método 
#zero padding, que consiste em adicionar uma camada de números "0" ao redor da imagem
#para não reduzir tanto o número de dados

#KERNEL(possuem praticamente a função de neurônios) é representado aqui como uma matriz de valores de uma parte da imagem que
#representa o resultado da multiplicação de cada um dos valores da matriz com seus
#respectivos pesos

#MAX POOLING é uma técnica utilizada para processar imagens grandes, baseia-se em
#escolher o maior valor de um kernel ao inves de multiplicar todos seus valores

#DROPOUT consiste em uma estratégia para evitar overfitting desligando alguns
#neurons aleatoriamente a uma taxa predeterminada, evitando overfitting

from tabnanny import verbose
import tensorflow.keras as keras
import pandas as pd

train_df = pd.read_csv('Data/sign_mnist_train.csv')
valid_df = pd.read_csv('Data/sign_mnist_valid.csv')

y_train = train_df['label']
y_valid = valid_df['label']

del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values

num_classes = 24

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)


x_train = x_train / 255
x_valid = x_valid / 255

#Nesse formato temos uma linha de 784 dígitos não sendo possível analisar os pixels
#assim é necessário formatar os dados de volta para a forma 28x28

#-1 indica a dimensão que não queremos alterar
x_train = x_train.reshape(-1, 28, 28, 1)
x_valid = x_valid.reshape(-1, 28, 28, 1)

#Utilizando um modelo de convolução utilizado em problemas similares

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

model = Sequential()
#CONV2D
#75 refere-se ao número de filtros que serão aprendidos, (3,3) refere-se ao tamanho
# dos filtros, STRIDES refere-se a magnitude das oscilções garantindo que o algoritmo
# se mantenha sempre próximo a esse valor, PADDING refere-se a quando a imagem de saída
# possui o mesmo tamanho da de entrada

model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", input_shape=(28, 28, 1)))
#BATCHNORMALIZATION normaliza a escala de valores das hidden layers
model.add(BatchNormalization())
#MAXPOOL2D pega uma imagem e a reduz para uma resolução menor, ajuda ao interpretar
# imagens com objetos em movimento

model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
#DROPOUT é a técnica utilizada para evitar overfitting ao desligar neurônios aleatórios
#tem como parâmetro a porcentagem de neurônios que serão desligados

model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
#FLATTEN pega a saídade de uma camada multidimensional e a transforma em uma camada
# unidimensional 

model.add(Flatten())
#DENSE essa primeira camada pega o vetor como input e o utiliza no processamento
# a segunda camada DENSE processa as saídas

model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=["accuracy"])

model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_valid, y_valid))

#RESULTADOS
#A precisão de treino está boa, no entanto a precisão de validação está mudando muito
# entre as epochs, o que significa que o modelo ainda não está generalizando
# muito bem