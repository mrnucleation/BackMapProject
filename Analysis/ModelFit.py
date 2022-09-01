import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from LoadData import loaddata, loadmodel
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

X_total, Y_total = loaddata()

X, X_test, Y, Y_test = train_test_split(X_total, Y_total, test_size=0.25)
#X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.25)

print(X.shape)
print(Y.shape)

model = loadmodel(X_total, Y_total)
print(X)

opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
#lossfunc = tf.keras.losses.MeanAbsoluteError()
lossfunc = tf.keras.losses.MeanSquaredError()
model.compile(
    optimizer=opt,
    loss=lossfunc
#    metrics=[tf.keras.metrics.Accuracy()]
    )

with open("testplot.dat", "w") as testfile:
    for i in range(100000):
        model.fit(X,Y,batch_size=30, epochs=5)
        Y_predict = model.predict(X_test)
        mae_test = tf.reduce_mean(tf.math.square(Y_predict-Y_test), axis=0)
        print("Model: %s, Score :%s"%(i, mae_test.numpy()))
        mae_test = tf.reduce_mean(mae_test, axis=0).numpy()
        model.save('backmap%s.model'%(i))
        testfile.write("%s %s\n"%(i, mae_test))
        testfile.flush()

