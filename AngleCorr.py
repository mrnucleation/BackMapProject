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
            "H": 'CH2',
            "He": 'AMD',
            "Li": 'CH3',
            "Be": 'CH',
            }

    atomspermol = 16
    outfilename = "cg_moldata.dat"
    if os.path.exists(outfilename):
        os.remove(outfilename)

    framenum = -1
    while True:
        framenum += 1
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

        #C3 - Atom next to AMD group - Atom Number - 4
        #C4 - Atom next to AMD group - Atom Number - 6
        #O1 - AMD Group 1 - Atom Number 15
        #O2 - AMD Group 1 - Atom Number 16
        #CH - CH Group - Atom Number 5
        subpairs = [
                  [15, 4], #Pair O1-C3
                  [15, 5], #Pair O1-CH
                  [16, 6], #Pair O2-C4
                  [16, 5] #Pair O2-CH
                ] 
        subtrips = [
                  [5, 15, 4], #Angle CH-O1-C3
                  [5, 16, 6] #Angle CH-O2-C4
                ] 


        moloutput = []
        for molid in range(nmols+1):
            molpairs = []
            for pair in subpairs:
                sub1, sub2 = tuple(pair)
                atm1 = molid*16+sub1-1
                atm2 = molid*16+sub2-1
                r = frame.get_distance(atm1, atm2, mic=True)*10.0
#                print(molid, atm1, atm2, r)
                molpairs.append(r)
            molangles = []
            for angle in subtrips:
                sub1, sub2, sub3 = tuple(angle)
                atm1 = molid*16+sub1-1
                atm2 = molid*16+sub2-1
                atm3 = molid*16+sub3-1
                ang = frame.get_angle(atm1, atm2, atm3, mic=True)
#                print(molid, atm1, atm2, atm3, ang)
                molangles.append(ang)
            moloutput.append( (molpairs,molangles) )
        with open(outfilename, "a") as outfile:
            outfile.write("Frame %s\n"%(framenum))
            outfile.write("#molid (N distances) (N angles)")
            for imol, molgeo in enumerate(moloutput):
                pairs, angles = molgeo
                outlist = [str(x) for x in pairs+angles]
                outstr = ' '.join(tuple(outlist))
                outfile.write("%s %s\n"%(imol, outstr))
            

#================================================
if __name__ == "__main__":
    main()

