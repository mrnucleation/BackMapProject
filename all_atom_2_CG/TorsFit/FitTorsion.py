import numpy as np
from sklearn.linear_model import LinearRegression


indata = np.loadtxt("CH2_CH2_CH2_O.dat")

phi = indata[:,0]
Y = indata[:,1]

print(phi)
X1 = (1.0+np.cos(phi))
X2 = (1.0-np.cos(2.0*phi))
X3 = (1.0+np.cos(3.0*phi))
X4 = (1.0-np.cos(4.0*phi))
X5 = (1.0+np.cos(5.0*phi))
X6 = (1.0-np.cos(6.0*phi))
X7 = (1.0+np.cos(7.0*phi))
#X1 = (1+np.cos(phi))
#X2 = (1-np.cos(2.0*phi))
#X3 = (1+np.cos(3.0*phi))
#X4 = (1-np.cos(4.0*phi))
#X5 = (1+np.cos(5.0*phi))
X = np.stack([X1,X2,X3,X4,X5,X6,X7], axis=1)

print(X.shape)
print(Y.shape)

linmodel = LinearRegression(fit_intercept=False)
linmodel.fit(X,Y)

Y_out = linmodel.predict(X)
print(linmodel.score(X, Y_out))
print(linmodel.coef_/(8.617e-5))

with open("outplot.dat", "w") as outfile:
    for x,y in zip(phi,Y_out):
        outfile.write("%s %s\n"%(x,y))



