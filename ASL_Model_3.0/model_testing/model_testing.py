#Objectives

# Load an already-trained model from disk
# Reformat images for a model trained on images of a different format
# Perform inference with new images, never seen by the trained model and evaluate its performance

#carregando o modelo

from tensorflow import keras

model = keras.models.load_model('../model/asl_model')

model.summary()

#Monstrando as novas imagens que serão utilizadas para testar o modelo

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')
    plt.show()


show_image('data/asl_images/a.png')


#Normalizando a imagem para o padrão do modelo, 28x28, em escalas de cinza, etc

from tensorflow.keras.preprocessing import image as image_utils

def loan_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode='grayscale',target_size=(28,28))
    return image

#Imprimindo a imagem para garantir que as normalização ocorreu

image = loan_and_scale_image('data/asl_images/a.png')
plt.imshow(image, cmap='gray')

#IMG_TO_ARRAY, converte a imagem em uma sequência de valores numéricos

image = image_utils.img_to_array(image)

image = image.reshape(1,28,28,1)
image = image / 255

#Fazendo previsões

prediction = model.predict(image)

print(prediction)

#O que temos como resultado é um array com 24(cada elemento representa a chance
# de ser uma das 24 letras do alfabeto) elementos representando probabilidades
# entre 0 e 1


#Interpretando o array de valores
import numpy as np

#ARMAX recebe um array e retorna seu maior valor de um eixto específico(nesse caso não especificado pois possui apenas um eixo)

#printa a previsão, nesse caso um valor representando a letra do alfabeto com maior probabilidade
print(np.argmax(prediction))

#Convertendo de número para letra do alfabeto
alphabet = "abcdefghiklmnopqrstuvwxy"

print("Prediction: ")
print(alphabet[np.argmax(prediction)])


#EXERCICIO, COLOCANDO TUDO ACIMA EM UMA FUNÇÃO

# def predict_letter(file_path):
#     #mostrando a imagem
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg
#     image = mpimg.imread(file_path)
#     plt.imshow(image, cmap='gray')
#     plt.show()
#     #colocando na escala de cor e tamanho corretos

#     from tensorflow.keras.preprocessing import image as image_utils
#     image = image_utils.load_img(file_path, color_mode='grayscale',target_size=(28,28))
#     plt.imshow(image, cmap='gray')
#     plt.show()
#     #Convertendo a imagem para um array
#     image = image_utils.img_to_array(image)
#     image = image.reshape(1,28,28,1)
#     image = image / 255
#     #Fazendo a previsão utilizando o modelo já feito

#     prediction = model.predict(image)
#     #Convertendo a previsão para uma letra

#     alphabet = "abcdefghiklmnopqrstuvwxy"
#     predicted_letter = alphabet[np.argmax(prediction)]
#     return predicted_letter

# predict_letter('data/asl_images/a.png')