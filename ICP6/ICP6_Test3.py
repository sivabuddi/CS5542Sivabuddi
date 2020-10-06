import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt

fileObject = open("myOutFile.txt", "r")
data = fileObject.read()
print(data)



# length of data is the number of characters in it
print ('Length of data: {} characters'.format(len(data)))

# Take a look at the first 250 characters in data
print(data[:250])

# The unique characters in the file
vocab = sorted(set(data))
print ('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

data_as_int = np.array([char2idx[c] for c in data])

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(data)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(data_as_int)



for i in char_dataset.take(5):
  print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
  input_data = chunk[:-1]
  target_data = chunk[1:]
  return input_data, target_data

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print(dataset)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(units=1024,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='orthogonal'),
    tf.keras.layers.GRU(units=512,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='orthogonal'),
    tf.keras.layers.GRU(units=128,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='orthogonal'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
#This gives us, at each timestep, a prediction of the next character index:
print(sampled_indices)


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def plot_diag(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy and loss')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.show()
example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='rmspop', loss=loss,metrics=['accuracy'])

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


EPOCHS=250
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
plot_diag(history)
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()

def generate_data(model, start_string):
  # Evaluation step (generating data using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  data_generated = []

  # Low temperatures results in more predictable data.
  # Higher temperatures results in more surprising data.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    data_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(data_generated))


print(generate_data(model, start_string=u"Akram "))

