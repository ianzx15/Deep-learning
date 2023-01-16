import keras
from keras.layers import Dense
from keras.models import Sequential
import pathlib
import PIL
import PIL.Image
import numpy as np
import tensorflow
import matplotlib.pylab as plt



train_dir = pathlib.Path("/home/ian/Documentos/Deep-learning/Deep-learning/archive/train")
valid_dir = pathlib.Path("/home/ian/Documentos/Deep-learning/Deep-learning/archive/valid")

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

class_names = train_ds.class_names

data_augmentation = keras.Sequential(
  [
    keras.layers.RandomFlip("horizontal",
                      input_shape=(224,
                                  224,
                                  3)),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
  ]
)

model = Sequential([
 data_augmentation,
  keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
  keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Dropout(0.2),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(450)
])
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=25
history = model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

img_path = "/home/ian/Documentos/Deep-learning/Deep-learning/archive/test/ABBOTTS BOOBY/1.jpg"

img = keras.utils.load_img(
    img_path, target_size=(224, 224)
)
img_array = tensorflow.keras.utils.img_to_array(img)
img_array = tensorflow.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tensorflow.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


model.save("birdsVision")

