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
    filename = sys.argv[1]
    framenum = -1
    model, model2 = loadlmodel()    
    while True:
        framenum += 1
        try:
            frame = read_lammps_dump(filename, index=framenum)
        except IndexError as e:
            print("End of File")
            break
        cgfeatures = computecg(frame)
        predictfeat = model.predict(cgfeatures)



#==================================================
def initialize(frame):
    positions = frame.get_positions()
    cell = frame.get_cell()
    atomtypes = frame.get_chemical_symbols()
    atomtypes = [typemap[atom] for atom in atomtypes]
    moltypes = []
    subid = 0
    subatomids = []
    molid = -1

    atomspermol = 16

    for i, atoms in enumerate(atomtypes):
        subid = i%atomspermol
        if subid == 0:
            molid += 1
        moltypes.append(molid)
        subatomids.append(subid)
    nmols = moltypes[-1]

    
 #===================================================================
def computecg(frame):
    typemap = {
            "H": 'CH2',
            "He": 'AMD',
            "Li": 'CH3',
            "Be": 'CH',
            }
    funcgroups = []
    atomtypes = frame.get_chemical_symbols()
    atomtypes = [typemap[atom] for atom in atomtypes]
    moltypes = []
    subid = 0
    subatomids = []
    molid = -1

    atomspermol = 16

    for i, atoms in enumerate(atomtypes):
        subid = i%atomspermol
        if subid == 0:
            molid += 1
        moltypes.append(molid)
        subatomids.append(subid)
    nmols = moltypes[-1]
    positions = frame.get_positions()

    funcgroups.append( (subpairs, subtrips, subquads) )

    moloutput = []
    for funcgroup in funcgroups:
        sp, st, sq = funcgroup
        molfeatures = []
        for molid in range(nmols+1):
            for pair in sp:
                sub1, sub2 = tuple(pair)
                atm1 = molid*atomspermol+sub1-1
                atm2 = molid*atomspermol+sub2-1
                r12 = frame.get_distance(atm1, atm2, mic=True)
                molfeatures.append(r12)

            for triplet in st:
                sub1, sub2, sub3 = tuple(triplet)
                atm1 = molid*atomspermol+sub1-1
                atm2 = molid*atomspermol+sub2-1
                atm3 = molid*atomspermol+sub3-1
                ang = frame.get_angle(atm1, atm2, atm3, mic=True)
                molfeatures.append(ang/180.0)

            for quad in sq:
                sub1, sub2, sub3, sub4 = tuple(quad)
                atm1 = molid*atomspermol+sub1-1
                atm2 = molid*atomspermol+sub2-1
                atm3 = molid*atomspermol+sub3-1
                atm4 = molid*atomspermol+sub4-1
                dihed = frame.get_dihedral(atm1, atm2, atm3, atm4, mic=True)
                molfeatures.append(ang/(360.0))

        moloutput.append(molfeatures)
    features = np.array(moloutput)
    return features
#===================================================================
def computestate(positions, nmols, targetfeat):
    psu_eng = np.float(0.0) 
    for molid in range(nmols+1):
        for pair in subpairs:
            sub1, sub2 = tuple(pair)
            atm1 = molid*atomspermol+sub1-1
            atm2 = molid*atomspermol+sub2-1
            r12 = computedist(positions, atm1, atm2)

        for triplet in subtrips:
            sub1, sub2, sub3 = tuple(triplet)
            atm1 = molid*atomspermol+sub1-1
            atm2 = molid*atomspermol+sub2-1
            atm3 = molid*atomspermol+sub3-1
            ang = computeangle(positions, atm1, atm2, atm3)

        for quad in subquads:
            sub1, sub2, sub3, sub4 = tuple(quad)
            atm1 = molid*atomspermol+sub1-1
            atm2 = molid*atomspermol+sub2-1
            atm3 = molid*atomspermol+sub3-1
            atm4 = molid*atomspermol+sub4-1
            ang = computetorsion(positions, atm1, atm2, atm3, atm4)


#================================================
if __name__ == "__main__":
    main()

