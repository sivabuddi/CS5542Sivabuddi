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

img_height = 180
img_width = 180
batch_size = 32

# Loading JSON file (Loading Architecture of existing one)
json_file = open("network_json.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.summary()


# Loading weights
loaded_model.load_weights("networks_weights.h5")

loaded_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)
#
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
#
# class_names = train_ds.class_names
# print(class_names)


sunflower_url = "/home/sivakumar/CS5542Sivabuddi/ICP5/sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('/home/sivakumar/CS5542Sivabuddi/ICP5/sunflower.jpg', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("The image most likely belongs to rose with {:.2f} percent confident".format(100 * np.max(score)))

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )









