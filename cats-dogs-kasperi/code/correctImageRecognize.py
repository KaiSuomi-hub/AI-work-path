from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model(".\\code\\KasperiMalli-1.keras")

# Load an image file to test, resizing it to the size your model was trained on
img = image.load_img('.\\Try\\kasperi.jpg', target_size=(224, 224))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a fourth dimension (since Keras expects a list of images)
x = np.expand_dims(x, axis=0)
print("vector ",x.shape)
print("x ",x)

# Normalize the input image's pixel values to the range used when training the neural network
x /= 512.#edit this
print("normalized ",x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)
print(predictions)
# The highest probability in the predictions list is the predicted class
predicted_class = np.argmax(predictions[0])

# Print the predicted class
if predicted_class == 0:
    print("This is an image of a generic dog.")
elif predicted_class == 1:
    print("This is an image of a generic dog.")
else:
    print("This is an image of a Kasperi.")