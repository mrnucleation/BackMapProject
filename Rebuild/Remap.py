import os
import numpy as np
import sys
import ase
from Model import ModelPipe
from GeoList import cg_geolist, aa_geolist, groupdata
from LoadData import loadmodel
from VectorFunc import computedist, computeangle, computetorsion, unittorsion, periodic
from random import randint, random
from math import log, pi

import jax.numpy as jnp
from jax import grad, jit

from ase.io.lammpsrun import read_lammps_dump



# ==================================================
def main():
    filename = sys.argv[1]
    framenum = -1
    outfile = 'outproc%s.lammpstrj'
    model = loadmodel()
    pipe = ModelPipe(model)
    while True:
        framenum += 1
        try:
            frame = read_lammps_dump(filename, index=framenum)
        except IndexError as e:
            print("End of File")
            break
        cell = frame.get_cell()
        cgfeatures = computecg(frame)
        predictfeat = pipe(cgfeatures)
        aa_positions, newtypes = preprocess(frame, predictfeat)
        outframe = ase.Atoms(newtypes, positions=aa_positions, cell=cell)
        outframe.wrap()
        outframe.write(outfile % (framenum), format='lammps-data')
        quit()


# ==================================================
def preprocess(frame, nn_features):
    # Sets up the data 
#    typemap = {
#        "H": 'CH2',
#        "He": 'AMD',
#        "Li": 'CH3',
#        "Be": 'CH',
#    }
    typemap = {
        "H": 'C',
        "He": 'AMD',
        "Li": 'C',
        "Be": 'C',
    }
    print(nn_features.shape)
    positions = frame.get_positions()
    cell = frame.get_cell().lengths()
#    print(cell)
    atomtypes = frame.get_chemical_symbols()
    atomtypes = [typemap[atom] for atom in atomtypes]
    moltypes = []
    subatomids = []
    molid = -1

    atomspermol = 16

    for i, atoms in enumerate(atomtypes):
        subid = i % atomspermol
        if subid == 0:
            molid += 1
        moltypes.append(molid)
        subatomids.append(subid)
    nmols = moltypes[-1]
    print("Number of Molecules:", nmols)

    newatompermol = atomspermol - 2 + 8  # Removing the two pseudo atoms and replacing it with all 4x2 real atoms
    natoms = newatompermol * nmols
#    print("newatompermol:", newatompermol)
#    print("natoms:", natoms)
    newcoords = []
    lb = 0
    ub = atomspermol 
    newtypes = []
    for i in range(nmols):
        molpos = positions[lb:ub, :]
        atm1 = np.copy(molpos[0, :])
        molpos[:,:] = molpos[:, :] - atm1
#        print(cell)
        for j, x in enumerate(molpos):
#            print(molpos[j])
            molpos[j] = np.where(molpos[j] > cell*0.5, molpos[j]-cell, molpos[j])
            molpos[j] = np.where(molpos[j] < -cell*0.5, molpos[j]+cell, molpos[j])
#            print(molpos[j])
        curtypes = atomtypes[lb:ub-2] + ['C', 'O', 'N', 'C', 'C', 'O', 'N', 'C']
        atomqueue, regrowdata = groupdata(nn_features[i, :])
#        print(curtypes)
        newatoms = np.zeros(shape=(6, 3))
        molpos = np.concatenate([molpos,newatoms], axis=0)
#        print(molpos.shape)
#        print()
#        print(molpos)
#        print()
#        print(atomqueue)
        while len(atomqueue) > 0:
#            print()
            nextatom = atomqueue.pop(0)

            # [17, 4, 3], 1.54, 109.5, torsion
            prevlist, r_bond, theta_bond, tors_angle = regrowdata[nextatom]
            theta_bond *= pi/180.0
            nextatom -= 1
            prevlist = [x-1 for x in prevlist]
            atomtype = curtypes[nextatom]
#            tors_angle = tors_list[i]
#            print(atomtype, nextatom, prevlist)
#            print(r_bond, theta_bond, tors_angle)
            v2 = molpos[prevlist[1], :] - molpos[prevlist[0], :]
#            v2 = periodic(v2, cell)
#            print("v2:",v2)
            v3 = molpos[prevlist[2], :] - molpos[prevlist[0], :]
#            print("v3:",v3)
#            v3 = periodic(v3, cell)
            vnew = unittorsion(v3, v2, r_bond, theta_bond, tors_angle)

#            print("v_new:",vnew)
#            print("molpos:", molpos[prevlist[0], :])
            molpos[nextatom, :] = vnew + molpos[prevlist[0], :]
            testang = computetorsion(molpos, nextatom, prevlist[0], prevlist[1], prevlist[2])
#            print("molpos_new:", molpos[nextatom, :])
#            print("New Angle:", testang)
        molpos = molpos + atm1
        newcoords.append(molpos)
#        with open("backmap.xyz", "w") as outfile:
#            outfile.write("%s \n" % (newatompermol+1))
#            outfile.write("\n")
#            outfile.write("C 0.0 0.0 0.0\n")
#            for x,t in zip(molpos, curtypes):
#                outfile.write("%s %s %s %s\n"%(t, x[0], x[1], x[2]))
#        molpos = relaxgeometry(molpos)
#        molpos = relaxgeometry2(molpos)
#        with open("postopt.xyz", "w") as outfile:
#            outfile.write("%s \n" % (newatompermol+1))
#            outfile.write("\n")
#            outfile.write("C 0.0 0.0 0.0\n")
#            for x,t in zip(molpos, curtypes):
#                outfile.write("%s %s %s %s\n"%(t, x[0], x[1], x[2]))

        newtypes = newtypes + curtypes
        lb += atomspermol
        ub += atomspermol
    newcoords = np.concatenate(newcoords, axis=0)

    print(newcoords.shape)

    return newcoords, newtypes


# ===================================================================
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
        subid = i % atomspermol
        if subid == 0:
            molid += 1
        moltypes.append(molid)
        subatomids.append(subid)
    nmols = moltypes[-1]
    positions = frame.get_positions()
    subpairs, subtrips, subquads, subpairs2, subtrips2, subquads2 = cg_geolist()
    head1 = (subpairs, subtrips, subquads)
    head2 = (subpairs2, subtrips2, subquads2)
    funcgroups.append(head1)
    funcgroups.append(head2)

    moloutput = []
    for funcgroup in funcgroups:
        sp, st, sq = funcgroup
        for molid in range(nmols):
            molfeatures = []
            for pair in sp:
                sub1, sub2 = tuple(pair)
                atm1 = molid * atomspermol + sub1 - 1
                atm2 = molid * atomspermol + sub2 - 1
                r12 = frame.get_distance(atm1, atm2, mic=True)
                molfeatures.append(r12)

            for triplet in st:
                sub1, sub2, sub3 = tuple(triplet)
                atm1 = molid * atomspermol + sub1 - 1
                atm2 = molid * atomspermol + sub2 - 1
                atm3 = molid * atomspermol + sub3 - 1
                ang = frame.get_angle(atm1, atm2, atm3, mic=True)
                molfeatures.append(ang / 180.0)

            for quad in sq:
                sub1, sub2, sub3, sub4 = tuple(quad)
                atm1 = molid * atomspermol + sub1 - 1
                atm2 = molid * atomspermol + sub2 - 1
                atm3 = molid * atomspermol + sub3 - 1
                atm4 = molid * atomspermol + sub4 - 1
                dihed = frame.get_dihedral(atm1, atm2, atm3, atm4, mic=True)
                molfeatures.append(dihed / (360.0))
            moloutput.append(molfeatures)
    features = np.array(moloutput)
    print("Features:", features.shape)
    return features
# ===================================================================
def relaxgeometry(molpos):
    curpos = np.copy(molpos)

    maxatom = 16 + 8 - 2 - 1
    ecur = computestate(molpos)
    for i in range(200):
        newpos = np.copy(curpos)
        natom = randint(14, maxatom)
        dr = 0.02 * np.random.uniform(-1.0, 1.0, size=(3))
#        print(natom, dr)
        newpos[natom,:] += dr
#        print(curpos - newpos)

        enew = computestate(newpos)
        ediff =  -(enew-ecur)/300.0 
        accept = False

        if  ediff >= 0.0: 
            accept = True
        elif  ediff >  log(random()) :
            accept = True
        if accept:
            print(enew, ecur)
            ecur = enew
            curpos = np.copy(newpos)
    return curpos

# ===================================================================
def relaxgeometry2(molpos):
    lrate = 8.4e-5
    curpos = jnp.array(molpos)

    for i in range(1000):
        test = computestate(curpos)
        gradient = grad(computestate)(curpos)
        gradient = gradient.at[:11,:].set(0.0)
        curpos -= lrate * gradient
        print(test)

        if test < 1e-1:
            break
    return np.array(curpos)
    
# ===================================================================
def computestate(positions):
    
    psu_eng = jnp.float64(0.0)
    forceconst = jnp.float64(1e3)
    angforceconst = jnp.float64(1e1)


#    psu_eng = 0.0
#    forceconst = 1e7
#    angforceconst = 1e5
    atomspermol = 16 + 8 - 2
    subpairs, subtrips, subquads, pair_eqs, trips_eqs = aa_geolist()
    trips_eqs = [x*np.pi/180.0 for x in trips_eqs]
#    print(psu_eng)
    for pair, eq in zip(subpairs, pair_eqs):
        sub1, sub2 = tuple(pair)
        atm1 = sub1 - 1
        atm2 = sub2 - 1
        r12 = computedist(positions, atm1, atm2)
        eng = forceconst * (r12 - eq) ** 2
#        print(atm1+1, atm2+1, r12, eng)
        psu_eng += eng

    for triplet, eq in zip(subtrips, trips_eqs):
        sub1, sub2, sub3 = tuple(triplet)
        atm1 = sub1 - 1
        atm2 = sub2 - 1
        atm3 = sub3 - 1
        ang = computeangle(positions, atm1, atm2, atm3) 
        eng = angforceconst * (ang - eq) ** 2
#        print(atm1+1, atm2+1, atm3+1, ang, eq, eng)
        psu_eng += angforceconst * (ang - eq) ** 2

#    print(psu_eng)
#       for quad, eq in zip(subquads, quad_eqs):
#           sub1, sub2, sub3, sub4 = tuple(quad)
#           atm1 = molid * atomspermol + sub1 - 1
#           atm2 = molid * atomspermol + sub2 - 1
#           atm3 = molid * atomspermol + sub3 - 1
#           atm4 = molid * atomspermol + sub4 - 1
#           ang = computetorsion(positions, atm1, atm2, atm3, atm4, cell)
#           psu_eng += forceconst * (ang - eq) ** 2
    return psu_eng


# ================================================
if __name__ == "__main__":
    main()
