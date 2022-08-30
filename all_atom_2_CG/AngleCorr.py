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
    #N1 - N-Butyl CH3 4nd Atom from AMD group - Atom Number - 1
    #C1 - N-Butyl CH2 3nd Atom from AMD group - Atom Number - 2
    #C2 - N-Butyl CH2 2nd Atom from AMD group - Atom Number - 3
    #C3 - N-Butyl CH2 1st Atom from AMD group - Atom Number - 4
    #CH - CH Group - Atom Number 5
    #C4 - N-Butyl CH2 1st Atom from AMD group - Atom Number - 6
    #C5 - N-Butyl CH2 2nd Atom from AMD group - Atom Number - 7
    #C6 - N-Butyl CH2 3rd Atom from AMD group - Atom Number - 8
    #N2 - N-Butyl CH3 4th Atom from AMD group - Atom Number - 9
    #C7 - Pentyl CH2 1st Atom from CH group - Atom Number - 10
    #O1 - AMD Group 1 - Atom Number 15
    #O2 - AMD Group 1 - Atom Number 16
    subpairs = [
#              [15, 4], #Pair O1-C3
#              [15, 5], #Pair O1-CH
              [16, 6], #Pair O2-C4
              [16, 5] #Pair O2-CH
            ] 
    subtrips = [
#              [5, 15, 4], #Angle CH-O1-C3
#              [10, 5, 15], #Angle C7-CH-O1
              [5, 16, 6], #Angle CH-O2-C4
              [10, 5, 16] #Angle C7-CH-O2
            ] 

#              [16, 5, 15], #Angle O2-CH-O1
#              [15, 4, 3], #Angle O1-C3-C2
#              [16, 6, 7] #Angle O2-C4-C5
    subquads = [
#            [15, 4, 3,2], #Angle O1-C3-C2-C1
#            [5, 15, 4, 3], #Angle CH-O1-C3-C2
#            [16, 5, 15, 6], #Angle O1-CH-O2-C4
            [16, 6, 7, 8], #Angle O2-C4-C5-C6
            [5 ,16, 6, 7], #Angle CH-O2-C4-C5
            [16, 5, 15, 4], #Angle O2-CH-O1-C3
            ] 
    nbins = 300
    dr = 3.5/float(nbins)
    dang = pi/float(nbins)
    ddihed = 2.0*pi/float(nbins)
    print(dr, dang, ddihed)
    disthist = np.zeros(shape=(len(subpairs), nbins))
    anglehist = np.zeros(shape=(len(subtrips), nbins))
    dihedhist = np.zeros(shape=(len(subquads), nbins))
    while True:
        framenum += 1
#        if framenum%10 != 0:
#            continue
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
#        positions = frame.get_positions()
#        positions *= 10.0 #Convert to Ang
        pairs = []
        angles = []

        moloutput = []

        for molid in range(nmols+1):
            molpairs = []
            for i, pair in enumerate(subpairs):
                sub1, sub2 = tuple(pair)
                atm1 = molid*16+sub1-1
                atm2 = molid*16+sub2-1
                r = frame.get_distance(atm1, atm2, mic=True)
                ibin = int(floor(10.0*r/dr))
#                print(ibin)
                if ibin < nbins:
                    disthist[i, ibin] += 1.0

                molpairs.append(r)
            molangles = []
            for i, angle in enumerate(subtrips):
                sub1, sub2, sub3 = tuple(angle)
                atm1 = molid*16+sub1-1
                atm2 = molid*16+sub2-1
                atm3 = molid*16+sub3-1
                ang = frame.get_angle(atm1, atm2, atm3, mic=True)
                ang *= pi/180.0
#                print(molid, atm1, atm2, atm3, ang)

                ibin = int(floor(ang/dang))
                if ibin < nbins:
                    anglehist[i, ibin] += 1.0
                molangles.append(ang/pi)
            moltors = []
            for i, angle in enumerate(subquads):
                sub1, sub2, sub3, sub4 = tuple(angle)
                atm1 = molid*16+sub1-1
                atm2 = molid*16+sub2-1
                atm3 = molid*16+sub3-1
                atm4 = molid*16+sub4-1
                dihed = frame.get_dihedral(atm1, atm2, atm3, atm4, mic=True)
                dihed *= pi/180.0
                ibin = int(floor(dihed/ddihed))
#                print(dihed, ddihed, ibin)
                if ibin < nbins:
                    dihedhist[i, ibin] += 1.0
                moltors.append(dihed/(2.0*pi))
            moloutput.append( (molpairs,molangles, moltors) )
        with open(outfilename, "a") as outfile:
            outfile.write("Frame %s\n"%(framenum))
            outfile.write("#molid (N distances) (N angles)\n")
            for imol, molgeo in enumerate(moloutput):
                pairs, angles, tors = molgeo
                outlist = [str(x) for x in pairs+angles+tors]
                outstr = ' '.join(tuple(outlist))
                outfile.write("%s %s\n"%(imol, outstr))
        for i, bond in enumerate(disthist):
            with open("hist%s.dat"%(i), "w") as outfile:
                for ibin, val in enumerate(bond):
                    r = dr*ibin
                    outfile.write("%s %s\n"%(r, val))
        for i, angle in enumerate(anglehist):
            with open("ang%s.dat"%(i), "w") as outfile:
                for ibin, val in enumerate(angle):
                    r = dang*ibin
                    outfile.write("%s %s\n"%(r, val))
        for i, dihed in enumerate(dihedhist):
            with open("dihed%s.dat"%(i), "w") as outfile:
                for ibin, val in enumerate(dihed):
                    r = ddihed*ibin
                    outfile.write("%s %s\n"%(r, val))




            

#================================================
if __name__ == "__main__":
    main()

