import sys
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from io import BytesIO
from keras.preprocessing import image
import keras
from tensorflow.keras.models import load_model
# Define the target size for the model
target_size = (224, 224)
image = Image.open(".\\Try\\ball.jpg")
def predict(img, top_n=3):
  img = Image.open(".\\Try\\ball.jpg")

  target_size = (180, 180)
  if img.size != target_size:
    img = img.resize(target_size)
  model = load_model('./code/KasperiMalli-1.keras')
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return decode_predictions(preds, top=top_n)[0]

def plot_preds(image, preds):

  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  order = list(reversed(range(len(preds))))
  bar_preds = [pr[2] for pr in preds]
  labels = (pr[1] for pr in preds)
  plt.barh(order, bar_preds, alpha=0.5)
  plt.yticks(order, labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


img = image
preds = predict(image)
plot_preds(img, preds)