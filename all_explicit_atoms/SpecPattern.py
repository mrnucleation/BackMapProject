import os
import sys
from math import sin, cos, pi, floor, sqrt
from scipy.fftpack import fftn, fftshift, ifftn
from scipy.signal import correlate
import numpy as np
import sys

from ase.io.lammpsrun import read_lammps_dump 
#==================================================
def main():
#    formfact = {
#            "H":  8,  #Actually CH2, but ASE returns Lammps type 1 as H
#            "He": 9,   #Actually CH3, but ASE returns Lammps type 2 as He
#            "Li": 30,   #Actually Amide, but ASE returns Lammps type 3 as He
#            "Be": 7   #Actually CH, but ASE returns Lammps type 4 as He
#            }
    formfact = {
            "H":  6,  #Actually C, but ASE returns Lammps type 1 as H
            "He": 7,   #Actually N, but ASE returns Lammps type 2 as He
            "Li": 8,   #Actually O, but ASE returns Lammps type 3 as He
            "B": 1   #Actually H, but ASE returns Lammps type 4 as He
            }
    filename = sys.argv[1]

    navg = 0
    framesToAvg = 15
    nBinsx = 100
    nBinsy = 100
    nBinsz = 100
    nx = nBinsx
    ny = nBinsy
    nz = nBinsz
    totBins = nBinsx * nBinsy * nBinsz
    bins = np.zeros(shape=( nBinsx, nBinsy, nBinsz))
    fAvg = np.zeros(shape=( nBinsx, nBinsy, nBinsz), dtype=np.complex128)

    nrbins = 200
    rbins = np.zeros(shape=( nrbins ))
    rcount = np.zeros(shape=( nrbins ))   
    framenum = -1
    with open("speckle.dump","w") as outfile:
        while True:
            framenum += 1
            bins = np.zeros(shape=( nBinsx, nBinsy, nBinsz))
            
            if navg > framesToAvg:
                rbins = np.zeros(shape=( nrbins ))
                rcount = np.zeros(shape=( nrbins ))   
                fAvg = np.zeros(shape=( nBinsx, nBinsy, nBinsz), dtype=np.complex128)
                nAvg = 0

            try:
                frame = read_lammps_dump(filename, index=framenum)
            except IndexError:
                print("End of File")
                break
            

            positions = frame.get_positions()
            positions *= 10.0 #Convert to Ang
            cell = frame.get_cell()
            cell *= 10.0 #Convert to Ang
            atomtypes = frame.get_chemical_symbols()
            print(atomtypes)
            Lx = cell[0][0]
            Ly = cell[1][1]
            Lz = cell[2][2]
            print(cell)
#            print(Lx, Ly, Lz)


            dx = nBinsx/Lx
            dy = nBinsy/Ly
            dz = nBinsz/Lz
            print(1.0/dx, 1.0/dy,1.0/dz)
#            print(atomtypes)

            for atomtype, atom in zip(atomtypes, positions):
                x, y, z = tuple(atom) 
#                print(atomtype, x, y, z)
                indx1 = int(floor(x*dx)) 
                indx2 = int(floor(y*dy)) 
                indx3 = int(floor(z*dz))

                try:
                    bins[indx1][indx2][indx3] += formfact[atomtype]
                except IndexError:
                    continue
            fTrans = fftshift(fftn(bins))
            navg += 1
#            ifTrans = ifftn(fTrans)
            fAvg += fTrans
            dr = 2.0/float(nrbins)

            print(fTrans.shape)
            dqx = 2.0*np.pi/Lx
            dqy = 2.0*np.pi/Ly
            dqz = 2.0*np.pi/Lz
            for index, c in np.ndenumerate(fAvg):
                nx,ny,nz = index
                try:
                    qx = dqx*(nx-50.0)
                    qy = dqy*(ny-50.0)
                    qz = dqz*(nz-50.0)
                except ZeroDivisionError:
                    continue
                    
                r = np.sqrt(qx**2 + qy**2 + qz**2)
#                print(qx,qy,qz, c)
                intens = abs(c)**2
                rbin = int(floor(r/dr))
#                print(dr)
                try:
                    rbins[rbin] += intens
                    rcount[rbin] += 1.0
                except IndexError:
                    continue
#                print(r, rbin, intens)
#            for intens, cnt in zip(rbins, rcount):
#                print(cnt, intens)
            half1 = nz // 2
            half2 = half1 // 2
            totBinsNew = nx*ny*nz

            if navg > framesToAvg:
                print("Printing to Frame")
                outfile.write("%s \n"%(nx*ny))
                outfile.write("\n")
                for index, val in np.ndenumerate(fAvg):
                    i, j, k = index
                    if k == half1:
                        outfile.write("%s %s %s %s \n"%(1, i, j, abs(val)/float(navg)))
                 #Radial Plot to File
                with open('radialplot.dump', "w") as outfile2:
                    outfile2.write("\n")
                    i = 0
                    for intens, cnt in zip(rbins, rcount):
                        i += 1
                        r = 0.5*dr*(i + (i-1))
                        if cnt < 1.0:
                            continue
                        avgintens = intens/cnt
                        outfile2.write("%s %s\n"%(r,avgintens))


#==================================================
class FormatError(Exception):
    pass
#================================================
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]
#================================================
if __name__ == "__main__":
    main()

