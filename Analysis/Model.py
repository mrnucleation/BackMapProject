import tensorflow as tf

class SimpleKeras(tf.keras.Model):

  def __init__(self, inlayer=7, outlayer=14):
    super().__init__()
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
    self.dense1 = tf.keras.layers.Dense(inlayer, activation=lrelu)
    self.dense2 = tf.keras.layers.Dense(62, activation=lrelu)
    self.dense3 = tf.keras.layers.Dense(62, activation=lrelu)
    self.dense4 = tf.keras.layers.Dense(62, activation=lrelu)
    self.dense5 = tf.keras.layers.Dense(outlayer, activation=tf.nn.relu)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    x = self.dense4(x)
    return self.dense5(x)


