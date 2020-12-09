import tensorflow.keras.layers
import tensorflow.keras.models
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


encoding_dim = 64
input_img =Input(shape=(784), name="encoder_input")

# BUILDING ENCODER
# "encoded" is the encoded representation of the input
encoded_dense = Dense(encoding_dim, kernel_initializer='random_uniform', activation='relu',name="Encoder_Dense_Layer_O")(input_img)
encoded_output = Dense(encoding_dim, kernel_initializer='random_uniform', activation='relu',name="Encoder_Active_Layer_O")(encoded_dense)
encoder = Model(input_img, encoded_output,name="Encoder_Model")
encoder.summary()

# BUILDING DECODER
# "decoded" is the encoded representation of the input
decoder_input = Input(shape=(encoding_dim),name='Decoder_Input')
decoding_dense = Dense(encoding_dim, kernel_initializer='random_uniform', activation='relu',name="Decoder_Dense_Layer_O")(decoder_input)
decoding_output = Dense(784, kernel_initializer='random_uniform', activation='relu',name="Decoder_Active_Layer_O")(decoding_dense)
decoder = Model(decoder_input, decoding_output,name="Decoder_Model")
decoder.summary()

# BUILD THE AUTO ENCODER i.e, establish the relation between input and reconstruction of an image
ae_input = Input(shape=(784), name="AE_Input")
ae_encoder_output = encoder(ae_input)
ae_decoder_output = decoder(ae_encoder_output)
ae_final = Model(ae_input, ae_decoder_output, name="Auto_Encoder")
ae_final.summary()



# AE Compilation
ae_final.compile(loss="mse", optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005),metrics='accuracy')
#ae_final.compile(optimizer='adadelta', loss='binary_crossentropy')
#ae_final.compile(optimizer='sgd', loss='binary_crossentropy')

# Preparing MNIST Dataset
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert 2-D array(28,28) to 1-D Array(784) for every image
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # (60,000, 784)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # (10,000, 784)

def plothist(hist,titlevalue='Accuracy without Data Augumentation'):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title(titlevalue)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


#Training AE
history = ae_final.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
plothist(history,"Epochs vs Loss")

num_images = 5
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)
noise = np.random.normal(loc=0.1, scale=0.1, size=x_test.shape)

#encoded_imgs = encoder.predict(x_test+noise)
decoded_imgs = ae_final.predict(x_test)




n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()





# num_images=5
# plt.figure(figsize=(18, 4))
# for i, image_idx in enumerate(random_test_images):
#     # plot original image
#     ax = plt.subplot(3, num_images, i + 1)
#     plt.imshow(x_test[image_idx].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # plot reconstructed image
#     ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
#     plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


# import numpy
# x_train = numpy.reshape(x_train_orig, newshape=(x_train_orig.shape[0], numpy.prod(x_train_orig.shape[1:])))
# x_test = numpy.reshape(x_test_orig, newshape=(x_test_orig.shape[0], numpy.prod(x_test_orig.shape[1:])))
#
# # Training AE
# ae_final.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
#
# encoded_images = encoder.predict(x_train)
# decoded_images = decoder.predict(encoded_images)
# decoded_images_orig = numpy.reshape(decoded_images, newshape=(decoded_images.shape[0], 28, 28))

# # RMSE
# def rmse(y_true, y_predict):
#     return tensorflow.keras.backend.mean(tensorflow.keras.backend.square(y_true-y_predict))
#
# # AE Compilation
# ae_final.compile(loss="mse", optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005))
#
# # Load the mnist dataset from keras
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # Normalize the dataset values in between 0 and 1
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.


# # Convert 2-D array(28,28) to 1-D Array(784) for every image
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # (60,000, 784)
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # (10,000, 784)
#
#
# #Training AE
# ae_final.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True, validation_data=(x_test, x_test))


#compile the file and fit the model
# ae_final.compile(loss="mse", optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005))
# ae_final.fit(x_train, x_train,epochs=5,batch_size=256,shuffle=True,validation_data=(x_test, x_test))


#
#
# # decoder
# decoded = Dense(784, activation='sigmoid')(encoded_active)
#
# # this model maps an input to its reconstruction
# autoencoder = Model(input_img, decoded)
#
#
# encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))
#
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
# from keras.datasets import mnist
# import numpy as np
#
# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# autoencoder.fit(x_train, x_train,
#                 epochs=5,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
#
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
# #decoded_imgs_new = autoencoder.predict(x_test)
#
# import matplotlib.pyplot as plt
#
# n = 4
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()