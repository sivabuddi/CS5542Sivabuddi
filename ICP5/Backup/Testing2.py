from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
import numpy as np

loaded_model = load_model('cifar10_overall_network.h5')
loaded_model
loaded_model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])


from keras.preprocessing import image
img_width=1000
img_height=667

test_image= image.load_img('/home/sivakumar/CS5542Sivabuddi/ICP5/cat.jpeg', target_size = (img_width,img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image.reshape(1000,667,-1)
result = loaded_model.predict(test_image)
print(result)
