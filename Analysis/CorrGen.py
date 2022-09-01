import numpy as np
from Model import SimpleKeras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from LoadData import loaddata, loadmodel
from math import pi

X_total, Y_total = loaddata()

#X, X_test, Y, Y_test = train_test_split(X_total, Y_total, test_size=0.25)

#model = SimpleKeras()
#model = tf.keras.models.load_model("backmap27.model")
model = loadmodel(X_total, Y_total)
print(Y_total.shape)

features = Y_total.shape[1]


Y_predict = model.predict(X_total)
err = tf.math.square(Y_predict-Y_total)
err = tf.reduce_mean(err, axis=0)
Y_total *= 360.0
Y_predict *= 360.0

print(err.numpy())

for feat in range(features):
    print(feat)
    with open("feat%s.dat"%(feat), "w") as trainfile:
        for y_targ, y_pred in zip(Y_total[:,feat], Y_predict[:,feat]):
            trainfile.write("%s %s\n"%(y_targ, y_pred))
