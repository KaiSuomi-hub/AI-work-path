# Purpose: Train a model using Keras and TensorFlow on the Kasperi dataset.
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense
import tensorflow as tf
import keras
import sys
# muokkaa 30 kun halutaan tarkkuutta
epochs=int(sys.argv[1])
################################################################
#hakemisto jossa kuvakansiot ovat
base_dir = '../PetImages/'

img_size = 224
batch = 64


# Create a data augmentor
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  validation_split=0.2)

# Create datasets
train_datagen = train_datagen.flow_from_directory(base_dir,
                                                  target_size=(
                                                      img_size, img_size),
                                                  subset='training',
                                                  batch_size=batch)
test_datagen = test_datagen.flow_from_directory(base_dir,
                                                target_size=(
                                                    img_size, img_size),
                                                subset='training',
                                                batch_size=batch)

# class_list = keras.utils.image_dataset_from_directory(base_dir, subset='training',validation_split=0.2,
# 	seed=2,
#     image_size=img_size,
#     batch_size=batch)

# class_names = class_list.class_names
# print("class_names")
# print(class_names)
# print("# of classes")
# num_classes = len(class_names)

# graphics when needed
# plt.figure(figsize=(10, 10))
# for images, labels in train_datagen.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")


# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same',
                 activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(3, activation="softmax"))

# Comment out if a model and parameter list are not needed.
# model.summary()

# keras.utils.plot_model(
#     model,
#     show_shapes = True,
#     show_dtype = True,
#     show_layer_activations = True
# )

# compile the model

model.compile(optimizer=tf.keras.optimizers.Adam(),loss='categorical_crossentropy', metrics=['accuracy'])

# tallennusnimi ja tiheys
callbacks = [
    keras.callbacks.ModelCheckpoint("KasperiMalli-{epoch}.keras"),
]
model.fit(train_datagen,epochs=epochs,callbacks=callbacks,validation_data=test_datagen)
