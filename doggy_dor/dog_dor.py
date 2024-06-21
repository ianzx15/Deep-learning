import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions


# load the VGG16 network "pre-trained model" on the ImageNet dataset
model = VGG16(weights="imagenet")

#This summary shows that the model expects images in the shape (224, 224, 3), colored images
model.summary()

def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)
    plt.show()

def load_and_process_image(image_path):
    
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(image_path).shape)

    # Load in the image with a target size of 224, 224
    image = image_utils.load_img(image_path, target_size=(224, 224))

    # Convert the image from a PIL format to a numpy array
    image = image_utils.img_to_array(image)

    # Add a dimension for number of images, in our case 1
    image = image.reshape(1,224,224,3)

    # Preprocess image to align with original ImageNet dataset
    image = preprocess_input(image)

    # Print image's shape after processing
    print(image)

    return image

processed_image = load_and_process_image("data/brown_bear.jpeg")


def readable_prediction(image_path):
    # Show image
    show_image(image_path)

    # Load and pre-process image
    image = load_and_process_image(image_path)

    # Make predictions
    predictions = model.predict(image)

    # Print predictions in readable form
    print('Predicted:', decode_predictions(predictions, top=3))

    #Index 151-268(dog), 281-285(cat)
    if 151 <= np.argmax(image) <= 268:
        print('DOG')
    elif 281 <= np.argmax(image) <= 285:
        print('CAT')
    else:
        print('OUT')

readable_prediction('data/brown_bear.jpeg')
