
def getpairs():
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
              [15, 4], #Pair O1-C3
              [15, 5], #Pair O1-CH
              [16, 6], #Pair O2-C4
              [16, 5] #Pair O2-CH
            ] 
    subtrips = [
              [5, 15, 4], #Angle CH-O1-C3
              [10, 5, 15], #Angle C7-CH-O1
              [5, 16, 6], #Angle CH-O2-C4
              [10, 5, 16] #Angle C7-CH-O2
            ] 

#              [16, 5, 15], #Angle O2-CH-O1
#              [15, 4, 3], #Angle O1-C3-C2
#              [16, 6, 7] #Angle O2-C4-C5
    subquads = [
            [15, 4, 3,2], #Angle O1-C3-C2-C1
            [5, 15, 4, 3], #Angle CH-O1-C3-C2
            [16, 5, 15, 6], #Angle O1-CH-O2-C4
            [16, 6, 7, 8], #Angle O2-C4-C5-C6
            [5 ,16, 6, 7], #Angle CH-O2-C4-C5
            [16, 5, 15, 4], #Angle O2-CH-O1-C3
            ] 
    return subpairs, subtrips, subquads
