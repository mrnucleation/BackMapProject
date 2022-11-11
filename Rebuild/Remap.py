import os
import numpy as np
import sys
import tensorflow as tf
import ase
from Model import ModelPipe
from GeoList import cg_geolist, aa_geolist, groupdata
from LoadData import loadmodel
from VectorFunc import computedist, computeangle, computetorsion, unittorsion, periodic


from ase.io.lammpsrun import read_lammps_dump 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
#==================================================
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
        cgfeatures = computecg(frame)
        predictfeat = pipe(cgfeatures)
        aa_positions, newtypes = preprocess(frame, predictfeat)
        outframe = ase.Atoms(newtypes, positions=aa_positions)
        outframe = frame.set_positions(aa_positions)
        outframe.wrap()
        outframe.write(outfile%(framenum), format='lammps-dump-text')


#==================================================
def preprocess(frame, nn_features):
    # Sets up the data 
    typemap = {
            "H": 'CH2',
            "He": 'AMD',
            "Li": 'CH3',
            "Be": 'CH',
            }
    positions = frame.get_positions()
    cell = frame.get_cell()
    atomtypes = frame.get_chemical_symbols()
    atomtypes = [typemap[atom] for atom in atomtypes]
    moltypes = []
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

    newatompermol = atomspermol - 2 + 8 #Removing the two pseudo atoms and replacing it with all 4x2 real atoms
    natoms = newatompermol*nmols
    print("natoms:", natoms)
    newcoords = [] 
    lb = 0
    ub = atomspermol - 1
    atomqueue, regrowdata = groupdata(nn_features)
    newtypes = []
    for i in range(nmols):
        molpos = positions[lb:ub, :]
        curtypes = moltypes[lb:ub]

        newatoms = np.zeros(shape=(8,3))
        molpos = np.concatenate([molpos] + newatoms, axis=0)
        while len(atomqueue) > 0:
            nextatom = atomqueue.pop(0)
            #[17, 4, 3], 1.54, 109.5, torsion
            prevlist, r_bond, theta_bond, tors_list = regrowdata[nextatom]
            tors_angle = tors_list[i]
            v2 = molpos[prevlist[1], :] - molpos[prevlist[0], :]
            v2 = periodic(v2, cell)
            v3 = molpos[prevlist[2], :] - molpos[prevlist[0], :]
            v3 = periodic(v3, cell)
            v1 = unittorsion(v3, v2, r_bond, theta_bond, tors_angle)
            molpos[nextatom, :] = v1 + molpos[prevlist[0], :]
        newcoords.append(molpos)
        lb += atomspermol
        ub += atomspermol
    newcoords = np.concatenate(newcoords, axis=0)

    print(newcoords.shape)

    return newcoords, newtypes
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
    print("Features:", features.shape)
    return features
#===================================================================
def computestate(positions, nmols, quad_eqs, cell):
    psu_eng = np.float(0.0) 
    forceconst = np.float(1e3) 

    atomspermol = 16 + 8 - 2
    subpairs, subtrips, subquads, pair_eqs, trips_eqs = aa_geolist()
    for molid in range(nmols+1):
        for pair, eq in zip(subpairs, pair_eqs):
            sub1, sub2 = tuple(pair)
            atm1 = molid*atomspermol+sub1-1
            atm2 = molid*atomspermol+sub2-1
            r12 = computedist(positions, atm1, atm2, cell)
            psu_eng += forceconst * (r12 - eq)**2

        for triplet, eq in zip(subtrips, trips_eqs):
            sub1, sub2, sub3 = tuple(triplet)
            atm1 = molid*atomspermol+sub1-1
            atm2 = molid*atomspermol+sub2-1
            atm3 = molid*atomspermol+sub3-1
            ang = computeangle(positions, atm1, atm2, atm3, cell)
            psu_eng += forceconst * (ang - eq)**2

        for quad, eq in zip(subquads, quad_eqs):
            sub1, sub2, sub3, sub4 = tuple(quad)
            atm1 = molid*atomspermol+sub1-1
            atm2 = molid*atomspermol+sub2-1
            atm3 = molid*atomspermol+sub3-1
            atm4 = molid*atomspermol+sub4-1
            ang = computetorsion(positions, atm1, atm2, atm3, atm4, cell)
            psu_eng += forceconst * (ang - eq)**2
#        print(molid, psu_eng)
    return psu_eng


#================================================
if __name__ == "__main__":
    main()

