import keras
import sys

from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model(sys.argv[1])
image_size = (224, 224)
img = keras.utils.load_img(sys.argv[2], target_size=image_size)
plt.imshow(img)
img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat, {100 * score:.2f}% Kasperi and {100 * (1-(0.5 * score)):.2f}% Dog.")
