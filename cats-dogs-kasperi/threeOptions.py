import keras
from keras import layers
import tensorflow as tf

image_size = (331,331)
batch_size = 32
train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    ".\\PetImages\\",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    batch_size=batch_size,
    image_size=image_size,
    color_mode =  "rgb",
    shuffle=True,
    seed=142,
    validation_split=0.2,
    subset= "training",
    interpolation="bilinear",
    follow_links=False,
)

validation_ds= tf.keras.preprocessing.image_dataset_from_directory(
    ".\\PetImages\\",
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=image_size,
     color_mode =  "rgb",
    shuffle=True,
    seed=142,
    validation_split=0.2,
    subset= "validation",
    interpolation="bilinear",
    follow_links=False,
    )
#Note: num_classes must be similar to the train_ds dataset 3 in this particular case.

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

train_ds = train_ds.prefetch(buffer_size=32)
validation_ds = validation_ds.prefetch(buffer_size=32)
model = tf.keras.applications.InceptionResNetV2(input_shape=(331,331, 3), include_top=False, weights='imagenet')

def make_model(input_shape, num_classes):
  inputs = keras.Input(shape=input_shape)
  # Image augmentation block
  x = data_augmentation(inputs)

  # Entry block
  x = layers.Rescaling(1./255)(x)
  x = model(x)

  previous_block_activation = x  # Set aside residual

  x = layers.GlobalAveragePooling2D()(x)
  if num_classes == 2:
    activation = 'sigmoid'
    units = 1
  else:
    activation = 'softmax'
    units = num_classes

  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(units, activation=activation)(x)
  return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=3)
keras.utils.plot_model(model, show_shapes=True)

epochs = 1
file_path = ".\model.keras"

checkpoint = tf.keras.callbacks.ModelCheckpoint("model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.keras", verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[checkpoint], verbose=2)