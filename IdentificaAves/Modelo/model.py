import keras
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.models import Sequential
import pathlib
import PIL
import PIL.Image
import numpy as np


# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)

# train_ds = keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(180, 180),
#   batch_size=32)

# val_ds = keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(180, 180),
#   batch_size=32)


# model = Sequential([
#   keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
#   keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Flatten(),
#   keras.layers.Dense(128, activation='relu'),
#   keras.layers.Dense(5)
# ])
# model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# epochs=10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )


train_path = "/home/ian/IdentificaAves/train"
valid_path = "/home/ian/IdentificaAves/valid"
train_dir = pathlib.Path("/home/ian/IdentificaAves/train/train")
valid_dir = pathlib.Path("/home/ian/IdentificaAves/valid")

train_ds = keras.utils.image_dataset_from_directory(train_dir,
  validation_split=0.2,
  subset="training",
  seed = 123,
  image_size=(224, 224),
  batch_size= 32)


valid_ds = keras.utils.image_dataset_from_directory(train_dir,
  validation_split=0.2,
  subset="validation",
  seed = 123,
  image_size=(224, 224),
  batch_size= 32) 


normalization_layer = keras.layers.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))



model = Sequential([
  keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(115)
])


model.compile(
  optimizer='adam',
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

epochs=10
history = model.fit(
  normalized_train_ds,
  validation_data=normalized_valid_ds,
  epochs=epochs
)

model.save("BirdsVision")




