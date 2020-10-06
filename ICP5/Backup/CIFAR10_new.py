import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
#from tensorflow.examples.tutorials.mnist import input_data
import keras as k
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import h5py
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

#load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_rows, img_cols , channels= 32,32,3


# set up image augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    #zoom_range=0.3
    )
datagen.fit(x_train)


#reshape into images
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, 1)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


#convert integers to float; normalise and center the mean
# x_train=x_train.astype("float32")
# x_test=x_test.astype("float32")
# mean=np.mean(x_train)
# std=np.std(x_train)
# x_test=(x_test-mean)/std
# x_train=(x_train-mean)/std

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# labels
num_classes=10
y_train = k.utils.to_categorical(y_train, num_classes)
y_test = k.utils.to_categorical(y_test, num_classes)


# plotting helper function
def plothist(hist,titlevalue='Accuracy without Data Augumentation'):
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title(titlevalue)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# build the model


#reg=l2(1e-4)   # L2 or "ridge" regularisation
reg=None
num_filters=32
ac='relu'
adm=Adam(lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt=adm
drop_dense=0.5
drop_conv=0

model = models.Sequential()

model.add(layers.Conv2D(num_filters, (3, 3), activation=ac, kernel_regularizer=reg, input_shape=(img_rows, img_cols, channels),padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))   # reduces to 16x16x3xnum_filters
model.add(layers.Dropout(drop_conv))

model.add(layers.Conv2D(2*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(2*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))   # reduces to 8x8x3x(2*num_filters)
model.add(layers.Dropout(drop_conv))

model.add(layers.Conv2D(4*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(4*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
model.add(layers.Dropout(drop_conv))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation=ac,kernel_regularizer=reg))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(drop_dense))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


#training without augmentation
history=model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
#training accuracy without dropout
train_acc=model.evaluate(x_train,y_train,batch_size=128)
print("Training Accuracy={}".format(train_acc))


test_acc=model.evaluate(x_test,y_test,batch_size=128)
print("Testing Accuracy={}".format(test_acc))

plothist(history)


model.save_weights("cifar10_without_DA_networks_weights.h5")
model.save("cifar10_without_DA_overall_network.h5")

# Saving model structure to a JSON file
model_json = model.to_json()

with open("cifar10_without_DA_network_json.json", "w") as json_file:
  json_file.write(model_json)


# build again, same model as model1

#reg=l2(1e-4)   # L2 or "ridge" regularisation
reg2=None
num_filters2=32
ac2='relu'
adm2=Adam(lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2=adm2
drop_dense2=0.5
drop_conv2=0

model2 = models.Sequential()

model2.add(layers.Conv2D(num_filters2, (3, 3), activation=ac2, kernel_regularizer=reg2, input_shape=(img_rows, img_cols, channels),padding='same'))
model2.add(layers.BatchNormalization(axis=-1))
model2.add(layers.Conv2D(num_filters2, (3, 3), activation=ac2,kernel_regularizer=reg2,padding='same'))
model2.add(layers.BatchNormalization(axis=-1))
model2.add(layers.MaxPooling2D(pool_size=(2, 2)))   # reduces to 16x16x3xnum_filters
model2.add(layers.Dropout(drop_conv2))

model2.add(layers.Conv2D(2*num_filters2, (3, 3), activation=ac2,kernel_regularizer=reg2,padding='same'))
model2.add(layers.BatchNormalization(axis=-1))
model2.add(layers.Conv2D(2*num_filters2, (3, 3), activation=ac2,kernel_regularizer=reg2,padding='same'))
model2.add(layers.BatchNormalization(axis=-1))
model2.add(layers.MaxPooling2D(pool_size=(2, 2)))   # reduces to 8x8x3x(2*num_filters)
model2.add(layers.Dropout(drop_conv2))

model2.add(layers.Conv2D(4*num_filters2, (3, 3), activation=ac2,kernel_regularizer=reg2,padding='same'))
model2.add(layers.BatchNormalization(axis=-1))
model2.add(layers.Conv2D(4*num_filters2, (3, 3), activation=ac2,kernel_regularizer=reg2,padding='same'))
model2.add(layers.BatchNormalization(axis=-1))
model2.add(layers.MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
model2.add(layers.Dropout(drop_conv2))

model2.add(layers.Flatten())
model2.add(layers.Dense(512, activation=ac2,kernel_regularizer=reg2))
model2.add(layers.BatchNormalization())
model2.add(layers.Dropout(drop_dense2))
model2.add(layers.Dense(num_classes, activation='softmax'))

model2.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

# train with image augmentation
history2=model2.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                    steps_per_epoch = len(x_train) / 128, epochs=10, validation_data=(x_test, y_test))

model2_train_acc=model2.evaluate(x_train,y_train,batch_size=128)
print("Training Accuracy with Data Augumentation={}".format(model2_train_acc))


model2_test_acc=model2.evaluate(x_test,y_test,batch_size=128)
print("Testing Accuracy with Data Augumentation={}".format(model2_test_acc))




plothist(history2,titlevalue="Accuracy with Data Augumentation")  # 128 batch, 0.001 lr, 512 neurons, no zoom, no convdrop, only 0.1 shift

model2.save_weights("cifar10_with_DA_networks_weights.h5")
model2.save("cifar10_with_DA_overall_network.h5")

# Saving model structure to a JSON file
model_json = model2.to_json()

with open("cifar10_with_DA_network_json.json", "w") as json_file:
  json_file.write(model_json)

