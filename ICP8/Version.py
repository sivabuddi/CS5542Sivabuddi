import tensorflow as tf;
x = [[2.]];
print("Tensorflow Version", tf.__version__)
print("hello TF world, {}".format(tf.matmul(x, x)))