from ase.io.lammpsrun import read_lammps_dump
from ase.io.lammpsdata import read_lammps_data
import numpy as np
def main():

    filename_aa = 'debug_atom.lammpstrj_aa' 
    filename_bm = 'outproc0.lammpstrj'

    framenum = 0
    try:
        frame_bm = read_lammps_data(filename_bm, style='atomic')
        frame_aa = read_lammps_dump(filename_aa, index=framenum)
    except IndexError as e:
        print("End of File: %s"%(e))
        return

    atompermol_aa = 58
    atompermol_bm = 22
    bm_positions = frame_bm.get_positions()
    aa_positions = frame_aa.get_positions()
    print(bm_positions)
    print(aa_positions)
# ['C', 'O', 'N', 'C', 'C', 'O', 'N', 'C']
 #               [18,56,53,11], #CH-C(O1)-N1-CH2
 #               [56,53,11,8], #C(O1)-N1-CH2-CH2
 #               [55,56,53,11], #O1-C(O1)-N1-CH2
 #               [14,53,11,8], #CH3-N1-CH2-CH2
 #               [53,11,8,5], #N1-CH2-CH2-CH2

#                [18,54,58,20], #CH-C(O2)-N2-CH2
#                [54,58,20,23], #C(O2)-N2-CH2-CH2
#                [57,56,53,11], #O2-C(O2)-N2-CH2
#                [33,58,23,26], #CH3-N2-CH2-CH2
#                [58,20,23,26], #N2-CH2-CH2-CH2

    atom_pairs = [
            (56,15),
            (55,16),
            (53, 17),
            (14,18),

            (54,19),
            (57,20),
            (58, 21),
            (33,22)
            ]
    nmols = 1000
    distances = []
    for molid in range(nmols+1):
        for sub_aa, sub_bm in atom_pairs:
            atm_aa = molid*atompermol_aa + sub_aa - 1
            atm_bm = molid*atompermol_bm + sub_bm - 1
            r_aa = aa_positions[atm_aa, :]
            r_bm = bm_positions[atm_bm, :]
            r_off = np.linalg.norm(r_aa-r_bm)
            distances.append(r_off)

    distances = np.array(distances)
    plot_histogram(distances)

# =================================================
import matplotlib.pyplot as plt
def plot_histogram(data):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    plt.rc('font', **font)
    plt.hist(data, bins=100, range=(0,2) )
    plt.xlabel('Error (Angstroms)')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()


if __name__ == "__main__":
    main()

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
