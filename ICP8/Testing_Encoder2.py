import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model, Sequential

# Denoised Output, Reconstructed, input,
# plot loss and accuracy using the history object.

(x_train, _), (x_test, _) = mnist.load_data()

# Scales the training and test data to range between 0 and 1.
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

#x_train.shape, x_test.shape; converting 2-D array into 1-D array
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # 60000,784
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # 10000,784

def plot_fig(history):
    # plotting the loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



input_dim = x_train.shape[1] # column size of each record
encoding_dim = 64


compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

autoencoder = Sequential()
autoencoder.add(
    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
)
autoencoder.add(
    Dense(input_dim, activation='sigmoid')
)

autoencoder.summary()
input_img = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[0]
encoder = Model(input_img, encoder_layer(input_img))

encoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics='accuracy')
history = autoencoder.fit(x_train, x_train,
                          epochs=3,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

plot_fig(history)
num_images = 5
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)
noise = np.random.normal(loc=0.1, scale=0.1, size=x_test.shape)

#encoded_imgs = encoder.predict(x_test+noise)
decoded_imgs = autoencoder.predict(x_test+noise)


plt.figure(figsize=(18, 4))
for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

