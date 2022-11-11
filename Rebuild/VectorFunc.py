#import numpy as np
import jax.numpy as np

#=====================================================
def periodic(r, cell):
    r = np.where(r < -cell*0.5, r+cell, r)
    r = np.where(r > cell*0.5, r-cell, r)
    return r
#=====================================================
def computedist(r, atm1, atm2, cell):
    r12 = r[atm1,:]-r[atm2,:]
    r12 = periodic(r12, cell)
    dist = np.linalg.norm(r12)
    return dist
#=====================================================
def computeangle(r, atm1, atm2, atm3, cell):
    r12 = r[atm1,:]-r[atm2,:]
    r12 = periodic(r12, cell)

    r32 = r[atm3,:]-r[atm2,:]
    r32 = periodic(r32, cell)

    dot1 = np.dot(r32, r12)
    angle = np.arccos(dot1)
    return angle
#=====================================================
def computetorsion(r, atm1, atm2, atm3, atm4, cell):
    r21 = r[atm2,:]-r[atm1,:]
    r21 = periodic(r21, cell)

    r32 = r[atm3,:]-r[atm2,:]
    r32 = periodic(r32, cell)

    r43 = r[atm4,:]-r[atm3,:]
    r43 = periodic(r43, cell)

    v1 = np.cross(r21, r32)
    v2 = np.cross(r32, r43)
    v3 = np.cross(v1, r32)
    dot1 = np.dot(v1, v2)
    dot2 = np.dot(v2, v3)
    angle = np.arctan2(dot2, dot1)
    angle += np.pi
    if np.isnan(angle):
        raise ValueError("NaN detected in torsion calculation")
    return angle
#=====================================================
def vectorgen(v1, r2, bond_ang, phi):
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
#===============================================================
def unittorsion(v1,v2,r3,bond_ang,tors_angle):
    #Code is converted from Fortran
#    r2 = v2[1]*v2[1] + v2[2]*v2[2] + v2[3]*v2[3]
#    r2 = sqrt(r2]
    r2 = np.linalg.norm(v2)
    r_proj = np.linalg.norm(v2[0:2])
#    r_proj = sqrt(v2(1)*v2(1) + v2(2)*v2(2))

    v1_u = v1-v2

#    Calculate the v1 vector's w2 and w3 components in the new orthonormal framework.
#    We don't need the x1_s as we'll be effectly ignoring that axis.
#    x1_s =  ( v2(1)*x1_u + v2(2)*v1_u + v2(3) * z1_u)/(r2)
    y1_s =  (-v2[1]*v1_u[0] + v2[1]*v1_u[1]) / r_proj
    z1_s =  (-v2[0]*v2[2]*v1_u[0] - v2[1]*v2[2]*v1_u[1] + r_proj*r_proj*v1_u[2]) / (r_proj*r2)
    #Calculate the torsional rotation angle for the new v3 vector from the v1 components
    rot_angle = np.atan2(z1_s, y1_s)
    rot_angle = rot_angle + tors_angle

    #Rescale the angle between (0, 2pi)
    while rot_angle < 0.0:
       rot_angle = rot_angle + 2.0*np.pi
    while rot_angle > 2.0*np.pi:
       rot_angle = rot_angle - 2.0*np.pi

    s_term = np.sin(rot_angle)
    c_term = np.cos(rot_angle)

    coeff1 = (r3/r2)*np.cos(bond_ang)
    coeff2 = (r3/r_proj)*np.sin(bond_ang)
    coeff3 = coeff2/r2

    v3 = np.zeros(size=3)
    v3[0] = coeff1*v2[0] - coeff2*c_term*v2[1] - coeff3*s_term*v2[0]*v2[2]
    v3[1] = coeff1*v2[1] + coeff2*c_term*v2[0] - coeff3*s_term*v2[1]*v2[2]
    v3[2] = coeff1*v2[2]                       + coeff3*s_term*(r_proj*r_proj)

    return v3
#=====================================================
