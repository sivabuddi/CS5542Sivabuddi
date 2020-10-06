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

# Loading JSON file (Loading Architecture of existing one)
json_file = open("cifar10_without_DA_network_json.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.summary()


# Loading weights
loaded_model.load_weights("cifar10_without_DA_networks_weights.h5")
loaded_model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])


import numpy as np
from keras.preprocessing import image
# Give the link of the image here to test
test_image1 =image.load_img('1car.jpg',target_size =(32,32))
test_image =image.img_to_array(test_image1)
test_image =np.expand_dims(test_image, axis =0)
result = loaded_model.predict(test_image)
print(result)
if result[0][0]==1:
    print("Aeroplane")
elif result[0][1]==1:
    print('Automobile')
elif result[0][2]==1:
    print('Bird')
elif result[0][3]==1:
    print('Cat')
elif result[0][4]==1:
    print('Deer')
elif result[0][5]==1:
    print('Dog')
elif result[0][6]==1:
    print('Frog')
elif result[0][7]==1:
    print('Horse')
elif result[0][8]==1:
    print('Ship')
elif result[0][9]==1:
    print('Truck')
else:
    print('Error')



from keras.preprocessing import image

'''
img_width=1000
img_height=667

test_image= image.load_img('/home/sivakumar/CS5542Sivabuddi/ICP5/cat.jpeg', target_size = (img_width,img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image.reshape(1000,667,-1)
result = loaded_model.predict(test_image)
score = tf.nn.softmax(predictions[0])
print("Score={}".format(100* np.max(score)))
'''


#... Use with images scraped from video (either GoPro or Android)
# from PIL import Image
# image = Image.open('sun.jpg').convert("RGB").resize((32, 32))
# img = np.array(image)
#
# r = img[:,:,0]
# g = img[:,:,1]
# b = img[:,:,2]
#
# npimages = np.array([[r] + [g] + [b]], np.uint8)
# npimages = npimages.transpose(0,2,3,1)
#
# classes = loaded_model.predict_classes(npimages)
# prediction = loaded_model.predict(npimages, verbose=2)
#
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
#
# predictions = loaded_model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# print("Score={}".format(100* np.max(score)))
#
# plt.imshow(img)