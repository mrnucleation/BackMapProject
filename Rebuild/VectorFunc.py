import numpy as np


#=====================================================
def computedist(r, atm1,atm2):
    r12 = r[atm1,:]-r1[atm2,:]
    dist = np.linalg.norm(r12)
    return dist
#=====================================================
def computeangle(r, atm1, atm2, atm3):
    r12 = r[atm1,:]-r1[atm2,:]
    r32 = r[atm3,:]-r1[atm2,:]
    dot1 = np.dot(r32, r12)
    angle = np.arccos(dot1)
    return angle
#=====================================================
def computetorsion(r, atm1, atm2, atm3, atm4):
    r12 = r[atm2,:]-r1[atm1,:]
    r32 = r[atm3,:]-r1[atm2,:]
    r43 = r[atm4,:]-r1[atm3,:]
    v1 = np.cross(r21, r32)
    v2 = np.cross(r32, r43)
    v3 = np.cross(v1, r32)
    dot1 = np.dot(v1, v2)
    dot2 = np.dot(v2, v3)
    angle = np.arctan2(dotv2, dotv1)
    return angle
#=====================================================
def vectorgen(v1, r2, bond_ang, phi, v2)
    r1 = np.linalg.norm(v1)

    s_term = np.sin(phi)
    c_term = np.cos(phi)      
    r_proj = np.linalg.norm(v1[0:2])
        
    coeff = np.zeros(size=3)
    coeff[0] = (r2/r1)*np.cos(bond_ang)
    coeff[1] = (r2/r_proj)*np.sin(bond_ang)
    coeff[2] = coeff[1]/r1

    w2 = np.zeros(size=3)
    w2[0] = -v1[1]
    w2[1] =  v1[0]
    w2 *= c_term

    w3 = np.zeros(size=3)
    w3[0] = -v1[0]*v1[2]
    w3[1] = -v1[1]*v1[2]
    w3[2] =  r_proj**2
    w3 *= s_term

    w_basis = np.stack([v1, w2, w3], axis=1)
    v2 = np.matmul(w_basis, coeff)

#    v2(1) = coeff1*v1(1) - coeff2*c_Term*v1(2) - coeff3*s_Term*v1(1)*v1(3)
#    v2(2) = coeff1*v1(2) + coeff2*c_term*v1(1) - coeff3*s_term*v1(2)*v1(3)
#    v2(3) = coeff1*v1(3)                       + coeff3*s_term*(r_proj*r_proj)
    return v2
#=====================================================