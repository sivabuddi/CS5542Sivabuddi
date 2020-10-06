from keras.models import load_model
import cv2
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from keras.preprocessing import image

img_height = 32
img_width = 32
batch_size = 128

# Loading JSON file (Loading Architecture of existing one)
json_file = open("cifar10_without_DA_network_json.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.summary()



# Loading weights
loaded_model.load_weights("cifar10_without_DA_networks_weights.h5")
loaded_model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])


sunflower_url = "/home/sivakumar/CS5542Sivabuddi/ICP5/sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('/home/sivakumar/CS5542Sivabuddi/ICP5/sunflower.jpg', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("prediction of cat is={}".format(100 * np.max(score)))

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )









