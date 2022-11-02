import tensorflow as tf
from keras import backend as K
import jax.numpy as np

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

class ModelPipe():
    def __init__(self, model):
        self.model = model
    def __call__(self, X_in):
        '''
        The data will have twice the data the model is trained for because there are
        two functional groups per molecule.  As such the data needs to be split into
        two equal sized blocks (one for each functional group per molecule) and fed in
        to the model separately.
        '''
        X_fragment = np.split(X_in, 2, axis=0)
        Y_out = []
        for xfrag in X_fragment:
            Y_frag = self.model(xfrag)
            Y_out.append(Y_frag.numpy())
        Y_out = np.concatenate(Y_out, axis=1)
        return Y_out
