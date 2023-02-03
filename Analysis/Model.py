import tensorflow as tf
from keras import backend as K


class SimpleKeras(tf.keras.Model):

  def __init__(self, inlayer=7, outlayer=14):
    super().__init__()
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
    self.layerlist = []
    for iVar in range(outlayer):
        layer = []
        dense1 = tf.keras.layers.Dense(inlayer+iVar, activation=lrelu)
        dense2 = tf.keras.layers.Dense(17, activation=lrelu)
        dense3 = tf.keras.layers.Dense(17, activation=lrelu)
        dense4 = tf.keras.layers.Dense(1, activation=tf.nn.tanh)
        layer.append([dense1, dense2, dense3, dense4])
        layerlist.append(layer)
    


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


