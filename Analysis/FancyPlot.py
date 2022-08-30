import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from collections import OrderedDict
from math import floor, exp, sqrt, ceil, log
import sys

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

linestyles = OrderedDict(
            [('solid',               (0, ())),
             ('loosely dotted',      (0, (1, 10))),
             ('dotted',              (0, (1, 5))),
             ('densely dotted',      (0, (1, 1))),
             ('loosely dashed',      (0, (5, 10))),
             ('dashed',              (0, (5, 5))),
             ('densely dashed',      (0, (5, 1))),
             ('loosely dashdotted',  (0, (3, 10, 1, 10))),
             ('dashdotted',          (0, (3, 5, 1, 5))),
             ('densely dashdotted',  (0, (3, 1, 1, 1))),
             ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
             ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
             ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

infile = sys.argv[1]
try: outfile = sys.argv[2]
except: outfile = None
indata = np.loadtxt(infile, dtype=float)
print(indata)
indata[:,0] = indata[:,0]/180.0
indata[:,1] = indata[:,1]/180.0
mine1 = indata[:,0].min() - 0.07
maxe1 = indata[:,0].max() + 0.07

mine2 = indata[:,1].min() - 0.07
maxe2 = indata[:,1].max() + 0.07
#mine1 = 0.0
#maxe1 = 1.0
#mine2 = 0.0
#maxe2 = 1.0
#if maxe1 > 1.0:
#    maxe1 = 1.0

#if mine1 < 0.0:
#    mine1 = 0.0

#if maxe1 > 0.:
mine1 = min(mine1, mine2)
maxe1 = max(maxe1, maxe2)
mine2 = mine1
maxe2 = maxe1
print( mine1, maxe1)
print( mine2, maxe2)



#x,y = np.meshgrid(np.linspace(mine1, maxe1, nbins), 
#                  np.linspace(mine2, maxe2, nbins))


nbins = 1000
gauss = int(round(nbins*0.01))
de1 = float(nbins)/(maxe1-mine1)
de2 = float(nbins)/(maxe2-mine2)
print( de1)
x = np.linspace(mine1,maxe1,nbins)
y = np.linspace(mine2,maxe2,nbins)

#factor = 0.3/0.2
#x = np.linspace(mine1*factor,maxe1*factor,nbins)
#y = np.linspace(mine1*factor,maxe1*factor,nbins)
xy = np.zeros(shape=(nbins,nbins), dtype=float)
for val1, val2 in indata:
#    print val1, val2
    if val1 > maxe1 or val1 < mine1:
        continue

    if val2 > maxe2 or val2 < mine2:
        continue
    nbin = int(floor((val1-mine1)*de1))
    nbin2 = int(floor((val2-mine2)*de2))
#    print nbin, nbin2
    for iBin in range(-gauss, gauss):
        for jBin in range(-gauss,gauss):
            if iBin**2 + jBin**2 > 1.5**2:
                continue
            xbin = nbin2+jBin
            ybin = nbin+iBin
            if xbin < 0 or ybin < 0:
                continue
            try:
                xy[xbin][ybin] += 1.0*exp(-(iBin**2 - jBin**2)/4e2)
            except IndexError:
                continue

xymin = xy.min()
xymax = xy.max()

for index, val in np.ndenumerate(xy):
    if val < 0.5:
        xy[index] = np.nan


linex = [mine1, maxe1]
liney = [mine2, maxe2]
#print linex, liney

plt.rc('font',**font)
plt.pcolormesh(x, y, xy, cmap=plt.cm.jet, norm=colors.LogNorm())
cbar = plt.colorbar(label='Point Density')
cbar.set_ticks([1.0,1e1,1e2,1e3, 1e4])
cbar.set_label(label='Count',weight='bold', fontsize=24)
c = 1e4
plt.clim(1.0, c+1.0)
plt.xlabel("True Dihedral Value (Reduced) ",weight='bold', fontsize=24)
plt.ylabel("Model Prediction (Reduced)",weight='bold', fontsize=24)
dx = (max(x)-min(x))/5.0
dy = (max(y)-min(y))/5.0

#plt.ylabel("Neural Network Force [eV/Angstrom]", fontsize=18, weight='bold')
#plt.xlabel("Reference Force [eV/Angstrom]", fontsize=18, weight='bold')
plt.xticks(np.arange(min(x), max(x), dx))
plt.yticks(np.arange(min(y), max(y), dy))

plt.plot(linex, liney, linestyle=linestyles["dashed"], color='black', linewidth=2.0)
plt.show()
if outfile is not None:
    plt.savefig(outfile, bbox_inches='tight', format='png', dpi=3000)
