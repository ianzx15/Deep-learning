import keras
import tensorflow as tf
import numpy as np
import pathlib


model = keras.models.load_model("/home/ian/Documentos/Deep-learning/Deep-learning/IdentificaAves/Modelo/birdsVision")

train_dir = pathlib.Path("/home/ian/Documentos/Deep-learning/Deep-learning/archive/train")
train_ds = keras.utils.image_dataset_from_directory(train_dir,
  validation_split=0.1,
  subset="training",
  seed = 123,
  image_size=(224, 224),
  batch_size= 32)
class_names = train_ds.class_names

img_path = "/home/ian/Documentos/Deep-learning/Deep-learning/archive/valid/ABYSSINIAN GROUND HORNBILL/2.jpg"

print(class_names)
img = keras.utils.load_img(
    img_path, target_size=(224, 224)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(score)
print("Especies: {}. Probability: {:.2f}".format(class_names[np.argmax(score)], 100 * np.max(score)))