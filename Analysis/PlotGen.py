import numpy as np
X = np.loadtxt("cg_moldata.dat")
Y = np.loadtxt("aa_moldata.dat")

X = X[:,1:]
#X[:,0:2] *= 1.0/0.35

Y = Y[:,1:]
#Y[:,0:5] *= 1.0/0.35


with open("corr.dat", "w") as outfile:
    for x,y in zip(X[:,-3], Y[:,-2]):
        outfile.write("%s %s\n"%(x,y))
