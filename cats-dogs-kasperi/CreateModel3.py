
import matplotlib.pyplot as plt
import numpy as np
# for displaying images
from PIL import Image

import tensorflow as tf
# for paths
from pathlib import Path

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# How many epochs to train for
epochCount = 11
data_dir = Path('.\\PetImages\\')
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Image count is ", image_count)

cats = list(data_dir.glob('Cat/*'))
dogs = list(data_dir.glob('Dog/*'))
kasperi = list(data_dir.glob('Kasperi/*'))

# Uncomment to check if the paths are correct.
# Should show you your img viewer and display the image.
# img=Image.open(str(cats[1]))
# img.show()

batch_size = 128
img_height = 180
img_width = 180
image_size = (img_height, img_width)
"""
Here's a breakdown of the parameters:

"data_dir": This is the path to the directory where the images are stored.
The function expects the images to be organized in subdirectories, where each subdirectory represents a class.

validation_split=0.2: This parameter specifies that 20% of the data should be reserved for validation.
Validation data is a subset of the training data used to evaluate the model during training and adjust its parameters. It's separate from the test data, which is used to evaluate the model after training.

subset="all": This parameter specifies which part of the data to load. The options are "training", "validation", or "all".
In this case, it's set to "all", which means both training and validation data will be loaded.

seed=1337: This is the random seed for shuffling the dataset. A random seed is a starting point for a sequence of random numbers.
Using the same seed will produce the same sequence of random numbers, which can be useful for reproducibility.

image_size=image_size: This parameter specifies the size to which the images will be resized after they are loaded.
The image_size variable is not defined in the selected code, but it would typically be a tuple specifying the height and width (in pixels).

batch_size=batch_size: This parameter specifies the number of samples to include in each batch.
The batch_size variable is not defined in the selected code, but it would typically be an integer.
When training a model, the data is usually processed in batches to make the computation more efficient.

The function returns two datasets: train_ds and val_ds. These are TensorFlow Dataset objects that can be used to train a model.
The train_ds dataset is used for training the model, and the val_ds dataset is used for validation.
"""
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
	subset="all",
     seed=2,
    image_size=image_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
print("class_names found")
print(class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
num_classes = len(class_names)
print("number of classes",num_classes)
print("model data variations")
########################################################################
##Data augmentation for a larger simulated dataset
"""
The code below is a function that applies data augmentation to an input set of images.
Specifically, it applies a random horizontal flip and a random rotation with an angle between -10% and 10% of the image's original angle.
"""
########################################################################
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
print("model creation")
model = Sequential([
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(3, activation='relu')
])


print("model compile")


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


print("model compile done")
model.summary()
epochs=epochCount
print("Epochs in use ",epochs)

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.fit(
  train_ds,
  validation_data=val_ds,
	callbacks=callbacks,
  epochs=epochs
)
