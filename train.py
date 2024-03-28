
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

"""
Let's create a dataset and force  the pictures to be 180x180 pixels in dimensions.
"""
image_size = (180, 180)
batch_size = 128
##This function is used to load a dataset of images from a directory and prepare it for training a machine learning model.
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
"""
Here's a breakdown of the parameters:

"PetImages": This is the path to the directory where the images are stored.
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
########################################################################
##Visualization
########################################################################
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

########################################################################
##Data augmentation for a larger simulated dataset
"""
The code below is a function that applies data augmentation to an input set of images.
Specifically, it applies a random horizontal flip and a random rotation with an angle between -10% and 10% of the image's original angle.
"""
########################################################################
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]
# Apply `data_augmentation` to the training images.
# We simulate a larger data set with mixing the images

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")



"""
With this option, your data augmentation will happen on device, synchronously with the rest of the model execution,
meaning that it will benefit from GPU acceleration.
Note that data augmentation is inactive at test time, so the input samples will only be augmented during fit(), not when calling evaluate() or predict().
If you're training on GPU, this may be a good option.
"""
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)



def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)


        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 3:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=3)
keras.utils.plot_model(model, show_shapes=True)

"""
## Train the model
"""

epochs = 11

callbacks = [
    keras.callbacks.ModelCheckpoint("train-py-save_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
