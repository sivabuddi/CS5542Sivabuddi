from keras.layers import Input, Dense
from keras.models import Model
import tensorflow.keras.models
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

encoding_dim = 64
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, kernel_initializer='random_uniform', activation='relu')(input_img)
encoded1 = Dense(encoding_dim, kernel_initializer='random_uniform', activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(encoded1)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded,name="Auto_Encoder")

encoder = Model(input_img, encoded,name="Encoder_Model")

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input),name="Decoder_Model")

encoder.summary()
decoder.summary()
autoencoder.summary()

from tensorflow.keras.optimizers import Adam
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(loss="mse", optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005))
#autoencoder.compile(loss='mse',optimizer='sgd')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

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
