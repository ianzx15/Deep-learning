import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing import image as image_utils


model = keras.models.load_model('asl_model')
alphabet = "abcdefghiklmnopqrstuvwxy"

#Checking if the model looks ok
# model.summary()

#PREPARING THE IMAGE FOR THE MODEL:

#Checking the image
# def show_image(image_path):
#     image = mpimg.imread(image_path)
#     plt.imshow(image, cmap='gray')
#     plt.show()

# show_image('data/asl_images/b.png')
    
# 1) Scaling the images to the same format as in the dataset (28x28 pixels, grayscale),
# the image needs to be downsized
def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="grayscale", target_size=(28,28))
    return image

image = load_and_scale_image('data/asl_images/b.png')
plt.imshow(image, cmap='gray')
plt.show()

# 2) Now the image needs to be reshaped into the same shape used in the dataset
image = image_utils.img_to_array(image)
# This reshape corresponds to 1 image of 28x28 pixels with one color channel
image = image.reshape(1,28,28,1) 

# 3) And last the image should be normalized into values between 0-1
image = image / 255

# 4) Making predictions
prediction = model.predict(image)
print(f'The model predicted: {alphabet[np.argmax(prediction)]}')