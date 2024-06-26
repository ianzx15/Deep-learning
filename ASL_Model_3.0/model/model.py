
#Objectives

# Augment the ASL dataset
# Use the augmented data to train an improved model
# Save the well-trained model to disk for use in deployment

#Utilizando o método de data augmentation, serve para aumentar a variedade dos dados
# introduzidos no modelo, diminuindo o overfitting e melhorando a precisão do modelo


#Utilizando o mesmo modelo da aula anterior:

import tensorflow.keras as keras
import pandas as pd

# Load in our data from CSV files
train_df = pd.read_csv('sign_mnist_train.csv')
valid_df = pd.read_csv('sign_mnist_valid.csv')

# Separate out our target values
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

# Separate image vectors
x_train = train_df.values
x_valid = valid_df.values

# Turn our scalar targets into binary categories
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Normalize our image data
x_train = x_train / 255
x_valid = x_valid / 255

# Reshape the image data for the convolutional network
x_train = x_train.reshape(-1,28,28,1)
x_valid = x_valid.reshape(-1,28,28,1)

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
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#IMAGEDATAGENERATOR, é uma função que aceita diversos parâmetros para gerar alterar
# certas características da imagem, dessa forma aumentando a quantidade de dados, sem
# necessitar de novas imagens

datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False, # Don't randomly flip images vertically
)

#Printando as imagens para certificar que estão sendo alteradas
#Batch é um grupo de tamanho limitado, nesse caso formado por 32 imagens

import matplotlib.pyplot as plt
import numpy as np
batch_size = 32
img_iter = datagen.flow(x_train, y_train, batch_size=batch_size)

x, y = img_iter.next()
fig, ax = plt.subplots(nrows=4, ncols=8)
for i in range(batch_size):
    image = x[i]
    ax.flatten()[i].imshow(np.squeeze(image))
plt.show()

#Colocando o modelo no dataset de treino

datagen.fit(x_train)

#Compilando o modelo

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

#Treinando o modelo

#É necessário utilizar um limitador de tempo para cada epoch já que será passada
# uma quantidade infinita de dados

#STEPS_PER_EPOCH define o tempo de cada epoch
model.fit(img_iter,
          epochs=20,
          steps_per_epoch=len(x_train)/batch_size, # Run same number of steps we would if we were not using a generator.
          validation_data=(x_valid, y_valid))

#Resultados:
#A validação de precisão está maior e mais constante que o da última aula
# indicando que o modelo conseguiu aprender melhor com essa formatação de dados


#Salvando o modelo para uso futuro

model.save('asl_model')