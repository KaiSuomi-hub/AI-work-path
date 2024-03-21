import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import tensorflow as tf

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
# cSpell:ignore relu downsamples Kasperi underfitting Overfitting errun Crossentropy
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")



train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

###Let's build a model. The issue is we might face is that the tensorflow we installed is optimized for CPU.
###tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
###Can you figure out a way to run this on GPU?


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_visible_devices(physical_devices[0], 'GPU')
    # List all physical devices that are of type 'GPU'
    physical_devices = tf.config.list_physical_devices('GPU')

    # If a GPU is available, this will print a list of GPU devices. If not, it will print an empty list.
    print(physical_devices)

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
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)
"""
The previous_block_activation = x line is storing the output of the current block of layers,
which will be used later to create a residual connection.

The loop for size in [256, 512, 728]: is iterating over a list of filter sizes.
These sizes represent the number of filters in the convolutional layers of the network. Each iteration of the loop adds a series of layers to the network with the current filter size.

Inside the loop, two sets of layers are added to the network: an activation function (ReLU),
a separable 2D convolution layer, and a batch normalization layer. These layers are added twice in succession.

After these layers, a 2D max pooling layer is added to the network.
This layer downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an
input window for each channel of the input.

The residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
line is creating a residual connection. This line applies a 2D convolution operation to the output of the
previous block of layers (stored in previous_block_activation), which is then added to the
output of the current block of layers (x) in the next line.

The x = layers.add([x, residual]) line is adding the residual connection to the output of the current block of layers.
This is done by adding the output of the residual connection (residual) to the output of the current block of layers (x).

The previous_block_activation = x line is updating the stored output of the previous block of layers
to be the output of the current block of layers (after the residual connection has been added).
This output will be used as the input to the next block of layers in the next iteration of the loop.

After the loop, a final set of layers is added to the network: a separable 2D convolution layer,
a batch normalization layer, and an activation function (ReLU).

The x = layers.GlobalAveragePooling2D()(x) line is adding a global average pooling layer to the network.
This layer computes the average of its input values over all dimensions, resulting in a single value per feature map.

Finally, the if num_classes == 2: units = 1 else: units = num_classes lines are setting the number of
units in the final layer of the network based on the number of classes in the data. If there are only two classes,
the final layer will have one unit (since a single output can represent two classes), otherwise, the final layer will have a
number of units equal to the number of classes.
"""

model = make_model(input_shape=image_size + (3,), num_classes=2)
"""
The make_model function takes two arguments: input_shape and num_classes.

The input_shape argument is expected to be a tuple that specifies the shape of the input data. In this case, image_size + (3,)
is passed as the input_shape. Here, image_size is a tuple that specifies the height and width of the input images, and (3,) is a
tuple containing a single element. When these two tuples are added together, the result is a new tuple that
includes the number of color channels in the images (3 for red, green, and blue). So, if image_size was (150, 150), input_shape would be (150, 150, 3).

The num_classes argument is the number of target classes in the data. This would typically be the number of categories that the CNN is
being trained to classify. In this case, num_classes is set to 3, indicating that there are three categories, Cats, Dogs and Kasperi.
"""
keras.utils.plot_model(model, show_shapes=True)

"""
## Train the model
"""

# epochs = 25
"""
This is an important value, epochs = 25 means that the training process will go through the entire dataset 25 times.

During each epoch, the model's weights are updated to minimize the loss function. The number of epochs is a hyperparameter that
defines the number times the learning algorithm will work through the entire training dataset.

Choosing the right number of epochs is important. Too few epochs can result in underfitting of the model,
while too many epochs can result in overfitting. Underfitting occurs when the model does not learn enough from the training data,
resulting in poor performance.
Overfitting occurs when the model learns too well from the training data, to the point where it performs poorly on new, unseen data.
For the first try you should leave it at 1 to see if the data goes through the training.
You can always errun the training with a larger value later.
"""
epochs = 1
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
###We save the model every run in a binary with a running name like save_at_1.keras
###Let's compile, be ready to wait from 15 to 30 minutes depending on the cpu.
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
"""
The optimizer argument is used to specify the optimization algorithm that will be used to update the model's weights during training.
In this case, the Adam optimizer is used, which is a popular choice because it combines the advantages of two other extensions of stochastic
gradient descent. Specifically,Adam uses techniques called adaptive gradient algorithm (AdaGrad)
and root mean square propagation (RMSProp). AdaGrad adapts the learning rate to the parameters, performing larger updates for
infrequent and smaller updates for frequent parameters. RMSProp also adapts the learning rates and works well in online and non-stationary settings.
The argument 3e-4 passed to keras.optimizers.Adam is the learning rate for the optimizer. This value is often between 0.0 and 1.0.
The learning rate controls how much to update the weight at the end of each batch and the smaller the value, the slower we travel along the downward slope.

The loss argument is used to specify the loss function that the model will try to minimize during training. In this case,
the Binary Cross-Entropy loss is used, which is suitable for binary classification problems. The from_logits=True argument means
that the function should interpret the model's output as raw scores (also known as logits). If from_logits is set to False,
the function would expect the model's output to be probabilities.
"""

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
"""
The fit method is called with four arguments: train_ds, epochs, callbacks, and validation_data.

The train_ds argument is the training data that the model will learn from. It is typically a tuple of Numpy arrays or a
dataset object that yields batches of inputs and targets. In this case, train_ds is presumably a tf.data.Dataset object that yields
batches of images and their corresponding labels.

The epochs argument is the number of epochs to train the model. An epoch is an iteration over the entire train_ds dataset.
This value is set to the epochs variable defined earlier in the code.

The callbacks argument is a list of keras.callbacks.Callback instances. Callbacks are functions that can be applied at
certain stages of the training process, such as at the end of each epoch. Typically, the list of callbacks is used to include
functionality such as model checkpointing (saving the model after each epoch), early stopping (stopping training when the validation
loss is no longer decreasing), and learning rate scheduling (changing the learning rate over time).

The validation_data argument is the data on which to evaluate the loss and any model metrics at the end of each epoch.
The model will not be trained on this data. This allows you to see how well the model generalizes to new data. In this case,
val_ds is presumably a tf.data.Dataset object that yields batches of validation images and their corresponding labels.
"""