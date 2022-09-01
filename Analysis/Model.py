import tensorflow as tf
from keras import backend as K


class SimpleKeras(tf.keras.Model):

  def __init__(self, inlayer=7, outlayer=14):
    super().__init__()
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
    self.dense1 = tf.keras.layers.Dense(inlayer, activation=lrelu)
    self.dense2 = tf.keras.layers.Dense(27, activation=lrelu)
    self.dense3 = tf.keras.layers.Dense(27, activation=lrelu)
#    self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.relu)
    self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.tanh)
    self.dense5 = tf.keras.layers.Dense(inlayer+1, activation=lrelu)
    self.dense6 = tf.keras.layers.Dense(25, activation=lrelu)
    self.dense7 = tf.keras.layers.Dense(25, activation=lrelu)
    self.dense8 = tf.keras.layers.Dense(outlayer-1, activation=tf.nn.tanh)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    x1 = self.dense4(x)
    x = tf.concat([inputs, x1], axis=1)
    x = self.dense5(x)
    x = self.dense6(x)
    x = self.dense7(x)
    x = self.dense8(x)
    x = tf.concat([x, x1], axis=1)

    return x


