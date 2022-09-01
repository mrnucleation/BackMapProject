import numpy as np
from Model import SimpleKeras
import tensorflow as tf
def loaddata():
    X1 = np.loadtxt("cg_moldata.dat")
    X1 = X1[:,1:]
    Y1 = np.loadtxt("aa_moldata.dat")
    Y1 = Y1[:,1:]
#    X2 = np.loadtxt("cg_moldata_group2.dat")
#    Y2  = np.loadtxt("aa_moldata_group2.dat")

#    X_total = np.concatenate([X1,X2],axis=0)
#    Y_total = np.concatenate([Y1,Y2],axis=0)
    X_total = X1
    Y_total = Y1
    print(X_total.shape)
    print(Y_total.shape)

    return X_total, Y_total

def loadmodel(X_total, Y_total):
    ninfeat = Y_total.shape[1]
    noutfeat = Y_total.shape[1]
#    model = SimpleKeras(inlayer=ninfeat, outlayer=noutfeat)
    model = tf.keras.models.load_model("backmap107.model")
    return model
