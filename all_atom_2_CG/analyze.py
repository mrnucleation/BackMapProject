import mdtraj
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

from matplotlib import rc
from matplotlib import rcParams


file_loc= '../../'
#traj = mdtraj.load(file_loc+'/Michael_trajs/pure_dmdbpma_gaff_final_config.gro', top = file_loc+'/Michael_trajs/pure_dmdbpma_gaff_final_config.gro')
trajAll = mdtraj.load_xtc(file_loc+'/Michael_trajs/pure_dmdbpma_gaff_traj.xtc', top = file_loc+'/Michael_trajs/pure_dmdbpma_gaff_final_config.gro', stride = 100)
topology = trajAll.topology
print(trajAll)
trajAll = trajAll[-50:]
print('File read completed')

def calc_com(atom_arr_amd):
    return np.array([np.sum(atom_arr_amd[:,0]*atom_arr_amd[:,3]), np.sum(atom_arr_amd[:,1]*atom_arr_amd[:,3]), np.sum(atom_arr_amd[:,2]*atom_arr_amd[:,3])]) / np.sum(atom_arr_amd[:,3])

def shift_com(com_amd, box_len):
    for j in range(3):
        com_amd[j] = com_amd[j] - np.floor(com_amd[j] / box_len[j]) * box_len[j]
    return com_amd

def output_dump_lammps_init(filename, box, header, data, fmt, frame_no):
    f2 = open(filename, 'a')
    f2.write(f'ITEM: TIMESTEP\n{frame_no}\n')
    f2.write('ITEM: NUMBER OF ATOMS\n%d\n' % data.shape[0])
    f2.write('ITEM: BOX BOUNDS pp pp pp\n')
    f2.write('%.7f %.7f\n' % (0, box[0]))
    f2.write('%.7f %.7f\n' % (0, box[1]))
    f2.write('%.7f %.7f\n' % (0, box[2]))
    f2.write('ITEM: ATOMS ' + header + '\n')
    f2.close()
    f2 = open(filename, 'ab')
    np.savetxt(f2, data, fmt=fmt)
    f2.close()
    return


debug = True
atm_typ = [1, 2, 3, 1, 5, 5, 5]
amd1 = ['C17','O1', 'N1', 'C5', 'H10', 'H11', 'H12']
amd2 = ['C18','O2', 'N2', 'C11', 'H23', 'H24', 'H25']
amd_union =list(set().union(amd1,amd2))
#typ_convert = {'C6':4, 'C1':3, 'C10':3, 'C16':3}
typ_convert = {'C2':1, 'C3':1, 'C4':1, 'C7':1, 'C8':1, 'C9':1, 'C12':1, 'C13':1, 'C14':1, 'C15':1, 'C6':4, 'C1':3, 'C10':3, 'C16':3}

#### NOTE: mdtraj units are in nm

for frame_no, traj in enumerate(trajAll):
    print(f'frame_id: {frame_no}')
    box_len = traj.unitcell_lengths
    id_k = 1
    id_list = []
    typ_list = []
    xyz_list = []
    res_list = []

    for residue in topology.residues:
        resindex = residue.index
        atom_arr_amd1 = np.zeros([1, 4])
        atom_arr_amd2 = np.zeros([1, 4])
        amd1_list_ind = [atom.index for atom in residue.atoms if atom.name in amd1]
        amd1_list_typ = [atm_typ[amd1.index(atom.name)] for atom in residue.atoms if atom.name in amd1]
        amd2_list_ind = [atom.index for atom in residue.atoms if atom.name in amd2]
        amd2_list_typ = [atm_typ[amd2.index(atom.name)] for atom in residue.atoms if atom.name in amd2]
        #new
        #other_list_ind = [atom.index for atom in residue.atoms if atom.name not in amd_union]
        #other_list_type = [atom.name[0] for atom in residue.atoms if atom.name not in amd_union]

        other_list_ind = [atom.index for atom in residue.atoms if (atom.name not in amd_union) and (atom.name [0]!='H')]
        other_list_type = [atom.name for atom in residue.atoms if (atom.name not in amd_union) and (atom.name [0]!='H')]

        for atom_idx in amd1_list_ind:
            atom_arr_amd1 = np.append(atom_arr_amd1, [np.hstack((traj.xyz[0, atom_idx,:], topology.atom(atom_idx).element.mass))], axis=0)

        for atom_idx in amd2_list_ind:
            atom_arr_amd2 = np.append(atom_arr_amd2, [np.hstack((traj.xyz[0, atom_idx,:], topology.atom(atom_idx).element.mass))], axis=0)

        atom_arr_amd1 = atom_arr_amd1[1:] 
        atom_arr_amd2 = atom_arr_amd2[1:] 
        #print(f'Done resindex = {resindex}')

        # shifting all atoms wrt to the first atom
        dx_amd1 = atom_arr_amd1[:,:3] - atom_arr_amd1[0,:3]
        dx_amd2 = atom_arr_amd2[:,:3] - atom_arr_amd2[0,:3]
        for j in range(3):
            dx_amd1[:,j] = np.rint(dx_amd1[:,j] / box_len[0, j]) * box_len[0,j]
            dx_amd2[:,j] = np.rint(dx_amd2[:,j] / box_len[0, j]) * box_len[0,j]
        atom_arr_amd1[:,:3] = atom_arr_amd1[:,:3] - dx_amd1
        atom_arr_amd2[:,:3] = atom_arr_amd2[:,:3] - dx_amd2

        # calculate com
        com_amd1 = calc_com(atom_arr_amd1)
        com_amd2 = calc_com(atom_arr_amd2)

        # shift com
        com_amd1 = shift_com(com_amd1,box_len[0,:])
        com_amd2 = shift_com(com_amd2,box_len[0,:])

        #create lists
        if debug == True: # printing all the constituent atoms too
            for j in range(len(other_list_type)):
                id_list.append(id_k)
                typ_list.append(typ_convert[other_list_type[j]])
                xyz_list.append(traj.xyz[0, other_list_ind[j],:])
                res_list.append(resindex)
                id_k+=1
        '''
        if debug: # printing all the constituent atoms too
            for j in range(len(atm_typ)):
                id_list.append(id_k)
                typ_list.append(amd1_list_typ[j])
                xyz_list.append(atom_arr_amd1[j,:3])
                res_list.append(resindex)
                id_k+=1
                id_list.append(id_k)
                typ_list.append(amd2_list_typ[j])
                xyz_list.append(atom_arr_amd2[j,:3])
                res_list.append(resindex)
                id_k+=1
        '''
        for j in range(1):
            id_list.append(id_k)
            typ_list.append(2)
            xyz_list.append(com_amd1)
            res_list.append(resindex)
            id_k+=1
            id_list.append(id_k)
            typ_list.append(2)
            xyz_list.append(com_amd2)
            res_list.append(resindex)
            id_k+=1

    xyz_list = np.array(xyz_list)
    output_dump_lammps_init(
        './debug_atom.lammpstrj',
        box_len[0],
        'id mol type x y z ',
        np.c_[
            id_list,
            res_list,
            typ_list,
            xyz_list[:, 0],
            xyz_list[:, 1],
            xyz_list[:, 2],
        ],
        '%d %d %d %.6f %.6f %.6f',
        frame_no,
    )




'''
# CH-CH3
list_indices = [atom.index for atom in topology.atoms if (atom.name=='C6')]
list_indices1 = [atom.index for atom in topology.atoms if (atom.name=='C16' or atom.name=='C10' or atom.name=='C1')]
pairs = topology.select_pairs(list_indices,list_indices1)
radii, rdf = mdtraj.compute_rdf(traj_reduced, pairs, r_range=[0,2], n_bins=100, periodic=True, opt=True)
np.savetxt('CH-CH3', (radii,rdf))
print('rdf1')


# CH3-CH3
list_indices = [atom.index for atom in topology.atoms if (atom.name=='C16' or atom.name=='C10' or atom.name=='C1')]
list_indices1 = [atom.index for atom in topology.atoms if (atom.name=='C16' or atom.name=='C10' or atom.name=='C1')]
pairs = topology.select_pairs(list_indices,list_indices1)
radii, rdf = mdtraj.compute_rdf(traj_reduced, pairs, r_range=[0,2], n_bins=100, periodic=True, opt=True)
np.savetxt('CH3-CH3', (radii,rdf))
print('rdf2')

# CH-CH
list_indices = [atom.index for atom in topology.atoms if (atom.name=='C6')]
list_indices1 = [atom.index for atom in topology.atoms if (atom.name=='C6')]
pairs = topology.select_pairs(list_indices,list_indices1)
radii, rdf = mdtraj.compute_rdf(traj_reduced, pairs, r_range=[0,2], n_bins=100, periodic=True, opt=True)
np.savetxt('CH-CH', (radii,rdf))
print('rdf3')
'''








