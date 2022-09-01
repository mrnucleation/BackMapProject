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
     # Since the lammps dumps have number based typing when you load this with ASE it mistakenly maps
     # the atomtypes according to the number of protons on the periodic table. This isn't correct
     # so to do this a type map is created to convert the atomtype list to the proper formatting.
    typemap = {
            "H": 'C',
            "He": 'O',
            "Li": 'N',
            "Be": 'CH',
            "B": 'H',
            }

    atomspermol = 58
    outfilename = "aa_moldata.dat"
    if os.path.exists(outfilename):
        os.remove(outfilename)

    framenum = -1
    while True:
        framenum += 1
        if framenum > 5:
            break
        try:
            frame = read_lammps_dump(filename, index=framenum)
        except IndexError as e:
            print("End of File")
#            print(e)
            break
#        print(frame)
#        print(dir(frame))

        atomtypes = frame.get_chemical_symbols()
        atomtypes = [typemap[atom] for atom in atomtypes]
        moltypes = []
        subid = 0
        subatomids = []
        molid = -1


        for i, atoms in enumerate(atomtypes):
            subid = i%atomspermol
            if subid == 0:
                molid += 1
            moltypes.append(molid)
            subatomids.append(subid)
        nmols = moltypes[-1]
        print(framenum, nmols)
        positions = frame.get_positions()
        positions *= 10.0 #Convert to Ang
        pairs = []
        angles = []
#        for atmtype, molid, subid, pos in zip(atomtypes, moltypes, subatomids, positions):
#            print(atmtype, molid, subid, pos)

         #C6- CH Group Carbon - AtmNum 18

         #C17 - Carbonyl Carbon - AtmNum 54
         #O1 - Carbonyl Oxygen next to C17 - AtmNum 55
         #N1 - Carbonyl Nitrogen next to C17 - AtmNum 53
         #C5- N-Methyl Group Carbon on N1 - AtmNum 14
         #C4- N-Butyl 1st Carbon from N1 - AtmNum 11
         #C3- N-Butyl 2st Carbon from N1 - AtmNum 8
         #C2- N-Butyl 3st Carbon from N1 - AtmNum 5
         #C1- N-Butyl 4st Carbon from N1 - AtmNum 1

         #C18 - Carbonyl Carbon - AtmNum 56
         #O2 - Carbonyl Oxygen next to C18 - AtmNum 57
         #N2 - Carbonyl Nitorgen next to C18 - AtmNum 58
         #C11 - N-Methyl Carbon On N2 - AtmNum 33
         #C7- N-Butyl 1st Carbon from N2 - AtmNum 20
         #C8- N-Butyl 2st Carbon from N2 - AtmNum 23
         #C9- N-Butyl 3st Carbon from N2 - AtmNum 26
         #C10- N-Butyl 4st Carbon from N2 - AtmNum 33
        subpairs = [
#                [18, 54], #C6-C17
#                [55, 54], #O1-C17
#                [53, 54], #N1-C17
#                [53, 14], #N1-C5
#                [11, 53], #N1-C4

#                [18, 56], #C6-C18
#                [57, 56], #O2-C18
#                [58, 56], #N2-C18
#                [58, 33], #N2-C11
#                [58, 20]  #N2-C7
                ] 
        subtrips = [
#                [18, 54, 55], #CH-C(O1)-O1
#                [18, 54, 53], #CH-C(O1)-N1
#                [55, 54, 53], #O1-C(O1)-N1

#                [54, 53, 14], #C(O1)-N1-CH3
#                [11, 53, 14], #CH2  -N1-CH3
#                [54, 53, 11], #C(O1)-N1-CH2

#                [18, 56, 55], #CH-C(O2)-O2
#                [18, 56, 58], #CH-C(O2)-N2
#                [57, 56, 58], #O2-C(O2)-N2

#                [56, 58, 33], #C(O2)-N2-CH3
#                [20, 58, 33], #CH2  -N2-CH3
#                [56, 58, 20]  #C(O2)-N2-CH2
                ] 
        subquads = [
                [18,56,53,11], #CH-C(O1)-N1-CH2
                [56,53,11,8], #C(O1)-N1-CH2-CH2
                [55,56,53,11], #O1-C(O1)-N1-CH2
                [14,53,11,8], #CH3-N1-CH2-CH2
                [53,11,8,5], #N1-CH2-CH2-CH2

#                [18,54,58,20], #CH-C(O2)-N2-CH2
#                [54,58,20,23], #C(O2)-N2-CH2-CH2
#                [57,56,53,11], #O2-C(O2)-N2-CH2
#                [33,58,23,26], #CH3-N2-CH2-CH2
#                [58,20,23,26], #N2-CH2-CH2-CH2
                ] 

        moloutput = []
        for molid in range(nmols+1):
            molpairs = []
            for pair in subpairs:
                sub1, sub2 = tuple(pair)
                atm1 = molid*58+sub1-1
                atm2 = molid*58+sub2-1
                r = frame.get_distance(atm1, atm2, mic=True)
#                print(molid, atm1, atm2, r)
                molpairs.append(r)
            molangles = []
            for triplet in subtrips:
                sub1, sub2, sub3 = tuple(triplet)
                atm1 = molid*58+sub1-1
                atm2 = molid*58+sub2-1
                atm3 = molid*58+sub3-1
                ang = frame.get_angle(atm1, atm2, atm3, mic=True)
                ang *= pi/180.0
#                print(molid, atm1, atm2, atm3, ang)
                molangles.append(ang/pi)
            moltors = []
            for quad in subquads:
                sub1, sub2, sub3, sub4 = tuple(quad)
                atm1 = molid*58+sub1-1
                atm2 = molid*58+sub2-1
                atm3 = molid*58+sub3-1
                atm4 = molid*58+sub4-1
                ang = frame.get_dihedral(atm1, atm2, atm3, atm4, mic=True)
                ang *= pi/180.0
#                print(molid, atm1, atm2, atm3, ang)
                moltors.append(ang/(2.0*pi))
            moloutput.append( (molpairs,molangles,moltors) )
        with open(outfilename, "a") as outfile:
            outfile.write("#Frame %s\n"%(framenum))
            outfile.write("#molid (N distances) (N angles)\n")
            for imol, molgeo in enumerate(moloutput):
                pairs, angles, tors = molgeo
                outlist = [str(x) for x in pairs+angles+tors]
                outstr = ' '.join(tuple(outlist))
                outfile.write("%s %s\n"%(imol, outstr))
            

#================================================
if __name__ == "__main__":
    main()

