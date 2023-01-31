from math import pi


#=======================================================
def cg_geolist():
    subpairs = [
              [15, 4], #Pair O1-C3
              [15, 5], #Pair O1-CH
            ] 
    subpairs2 = [
              [16, 6], #Pair O2-C4
              [16, 5] #Pair O2-CH
            ] 
    subtrips = [
              [5, 15, 4], #Angle CH-O1-C3
              [10, 5, 15], #Angle C7-CH-O1
            ] 
    subtrips2 = [
              [5, 16, 6], #Angle CH-O2-C4
              [10, 5, 16] #Angle C7-CH-O2
            ]
    subquads = [
            [15, 4, 3,2], #Angle O1-C3-C2-C1
            [5, 15, 4, 3], #Angle CH-O1-C3-C2
            [16, 5, 15, 6], #Angle O1-CH-O2-C4
            ] 
    subquads2 = [
            [16, 6, 7, 8], #Angle O2-C4-C5-C6
            [5 ,16, 6, 7], #Angle CH-O2-C4-C5
            [16, 5, 15, 4], #Angle O2-CH-O1-C3
            ]
    return subpairs, subtrips, subquads, subpairs2, subtrips2, subquads2


#=======================================================
def aa_geolist():
    #C1
    subpairs = [
              [15, 5], #Pair C(=O1)-CH
              [15, 16], #Pair C = O1
              [15, 17], #Pair C(=O1)-N
              [17, 18], #Pair N1-CH3
              [17, 4], #Pair N1-CH2(Butyl)

              [19, 5], #Pair C(=O2)-CH
              [20, 19], #Pair C = O2
              [21, 19], #Pair C(=O2)-N
              [21, 22], #Pair N2-CH3
              [21, 6], #Pair N2-CH2(Butyl)
            ]
    paireqs = [
            0.154,
            0.1204,
            0.134,
            0.154,
            0.134,

            0.154,
            0.1204,
            0.134,
            0.154,
            0.134,
            ]
    #15 = C
    #16 = O1
    #17 = N1
    #18 = CH3

    #19 = C
    #20 = O2
    #21 = N2
    #22 = CH3
    subtrips = [
            [5, 15, 16], #CH-C=O1
            [5, 15, 17], #CH-C(=O1)-N
            [17, 15, 16], #N1-C=O1
            [4, 17, 15], #CH2-N-C(=O1)
            [18, 17, 15], #CH3-N-C(=O1)
            [18, 17, 4], #CH3-N-CH2

            [5, 19, 20], #CH-C=O2
            [5, 19, 21], #CH-C(=O2)-N
            [21, 19, 20], #N1-C=O2
            [6, 21, 19], #CH2-N-C(=O2)
            [22, 21, 19], #CH3-N-C(=O2)
            [22, 21, 6], #CH3-N-CH2
            ] 
    trip_eqs = [
            121.0,
            123.0,
            125.0,
            123.0,
            123.0,
            114.0,

            121.0,
            123.0,
            125.0,
            123.0,
            123.0,
            114.0,
            ]

    subquads = [
            [5, 15, 17, 4], #CH-C(O1)-N1-CH2
            [15, 17, 4, 3], #C(O1)-N1-CH2-CH2
            [16, 15, 17, 4], #O1-C(O1)-N1-CH2
            [18, 17, 4, 3], #CH3-N1-CH2-CH2
            [17, 4, 3, 2], #N1-CH2-CH2-CH2

            [5, 19, 21, 6], #CH-C(O2)-N2-CH2
            [19, 21, 6, 7], #C(O2)-N2-CH2-CH2
            [20, 19, 21, 6], #O2-C(O2)-N2-CH2
            [22, 21, 6, 7], #CH3-N2-CH2-CH2
            [21, 6, 7, 8], #N2-CH2-CH2-CH2
            ] 
    return subpairs, subtrips, subquads, paireqs, trip_eqs

#[C, O, N, CH3, C, O, N, CH3]
#=======================================================
def groupdata(nn_features):
    #Atoms that need to be regrown: 15, 16, 17, 18, 19,20,21,22
    #Prefered order: 17, 15, 18, 16 // 21, 19, 22, 20
    print(nn_features)
    outqueue = [17, 15, 18, 16, 21, 19, 22, 20]
    '''
    featuretable = {
        15: ([17, 4, 3], 0.134, 109.5, nn_features[1]),
        16: ([15, 17, 4], 0.1204, 120.0,nn_features[2]),
        17: ([4, 3, 2], 0.134, 109.5,  nn_features[4]),
        18: ([17, 4, 3], 0.154, 109.5, nn_features[3]),

        19: ([21, 6, 7], 0.134, 109.5, nn_features[6]),
        20: ([19, 21, 6], 0.1204, 120.0,nn_features[7]),
        21: ([6, 7, 8], 0.134, 109.5,nn_features[9]),
        22: ([21, 6, 7], 0.134, 109.5, nn_features[8]),
    }

    '''
    #5 - CH

    #15 - C1(=O1)
    #16 - O1
    #17 - N1
    #4 - CH2-(N1)
    #18 - CH3-(N1)

    #19 - C2(=O2)
    #20 - O2
    #21 - N2
    #5 - CH2-(N2)
    #22 - CH3-(N2)

    featuretable = {
        15: ([17, 4, 3], 0.134, 115.0, nn_features[1]),  #C(=O)
        16: ([15, 17, 4], 0.1204, 120.0,nn_features[2]),  #O
        17: ([4, 3, 2], 0.134, 115.0, nn_features[4] ),   #N
        18: ([17, 4, 3], 0.154, 115.0, nn_features[3]),  #CH3

        19: ([21, 6, 7], 0.134, 115.0, nn_features[6]), #C(=O)
        20: ([19, 21, 6], 0.1204, 120.0, nn_features[7]), #O
        21: ([6, 7, 8], 0.134, 115.0, nn_features[9] ),  #N
        22: ([21, 6, 7], 0.154, 115.0, nn_features[8]), #CH3
    }
    return outqueue, featuretable
#=======================================================

