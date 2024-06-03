import IPython

import matplotlib.pyplot as plt

import tensorflow

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

# To check dataset images
# image = x_train[0]
# plt.imshow(image, cmap='gray')
# plt.show()

#Normalizando o formato para apenas uma linha com 784 caracteres (antes era uma matriz 28 x 28)
#O primeiro valor indica a quantidade de itens e o segundo indica o formato
x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)

#Normalizando todos os valores de 0 a 255 para 0 a 1
x_train = x_train / 255
x_valid = x_valid / 255

import tensorflow.keras as keras
#Normalizando o formato dos itens de comparação 'y' para código binário exclusivo do keras
num_categories = 10
#NUM_CATEGORIES indica a quantidade de caracteres que serão utilizados para representar o valor
y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)


from tensorflow.keras.models import Sequential
#Instancia o modelo que receberá as diversas camadas de dados(cria uma espécia
#  de molde ou casca para agrupar os dados que virão)

model = Sequential()

from tensorflow.keras.layers import Dense
#UNITS especifica o numero de neurônios na camada(pode ser alterado dependendo do objetivo)

#RELU é uma activation function(uma função que recebe valores e seus respectivos pesos e os soma 
# gerando uma única saída de dados não lineares para a próxima camada)

#INPUT_SHAPE indica o formato de dados que serão recebidos, nesse caso um array de uma dimensão

model.add(Dense(units=512, activation='relu', input_shape=(784,)))

#Adicionando mais uma camada de processamento
model.add(Dense(units=512, activation='relu'))

#Adicionando a camada de saída

#SOFTMAX é uma activation function que gera um valor probabilístico para cada camada variando de 0 a 1, sendo
# a soma total dos valores das camadas sempre igual a 1
# For example, softmax might determine that the probability of a
#  particular image being a dog at 0.9, a cat at 0.08, and a horse at 0.02. 

#UNITS cada unidade da camada de saída está relacionada com a forma de entrada dos dadaos, como
#cada dígito de cada linha da matriz que representa a matriz está representado em um número binário de 10 dígitos,
# dessa forma temos 10 neurônios avaliando 10 dígitos por vez
model.add(Dense(units = 10, activation='softmax'))

#Para printar um sumário do que há no modelo
model.summary()

#Antes de treinar o modelo é necessário compilar, para isso utiliza-se a loss function que 
#será utilizada para mostrar a perfomance durante o treino

#CATEGORICAL_CROSSENTROPY uma loss function que calcula a a diferença entre a distribuição de
#duas probabilidades

#ACCURACY define o que queremos acompanhar durante o treino dos modelos
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

#Treinando o modelo:
#FIT utilizando esse método é necessário passar os seguintes parâmtros:
#os dados utilizados, os dados de correção/comparação (y_train), quantas vezes
#deve treinar 'EPOCHS', e os dados de validação de x e y

#VERBOSE apenas muda como o dado é mostrado para o usuário, '0' não mostra nada '1' mostra barras
# '2' não mostra barras

history = model.fit(
    x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid)
)

