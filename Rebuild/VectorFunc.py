#import numpy as np
import jax.numpy as np

#=====================================================
def periodic(r, cell):
    r = np.where(r < 0.0, r+cell, r)
    r = np.where(r > cell, r-cell, r)
    return r
#=====================================================
def computedist(r, atm1, atm2):
    r12 = r[atm1,:]-r[atm2,:]
    dist = np.linalg.norm(r12)
    return dist
#=====================================================
def computeangle(r, atm1, atm2, atm3):
    r12 = r[atm1,:]-r[atm2,:]
#    r12 = periodic(r12, cell)

    r32 = r[atm3,:]-r[atm2,:]
#    r32 = periodic(r32, cell)

    dot1 = np.dot(r32, r12)
    dot1 /= (np.linalg.norm(r32) * np.linalg.norm(r12))
    angle = np.arccos(dot1)
    return angle
#=====================================================
def computetorsion(r, atm1, atm2, atm3, atm4):

    b1 = r[atm2,:] - r[atm1,:]
    b2 = r[atm3,:] - r[atm2,:]
    b3 = r[atm4,:] - r[atm3,:]


    # Define the normal vectors for the planes formed by the vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize the normal vectors
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    # Compute the angle between the normal vectors
    angle = np.arccos(np.dot(n1, n2))

    # Define the sign of the angle
    sign = np.sign(np.dot(np.cross(n1, n2), b2))

    return sign * angle + np.pi
#=====================================================
def computetorsion2(r, atm1, atm2, atm3, atm4):
    print(atm1, atm2, atm3, atm4)
    r21 = r[atm2,:]-r[atm1,:]
#    r21 = periodic(r21, cell)

    r32 = r[atm3,:]-r[atm2,:]
#    r32 = periodic(r32, cell)

    r43 = r[atm4,:]-r[atm3,:]
#    r43 = periodic(r43, cell)

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
        
    coeff = np.zeros(shape=3)
    coeff[0] = (r2/r1)*np.cos(bond_ang)
    coeff[1] = (r2/r_proj)*np.sin(bond_ang)
    coeff[2] = coeff[1]/r1

    w2 = np.zeros(shape=3)
    w2[0] = -v1[1]
    w2[1] =  v1[0]
    w2 *= c_term

    w3 = np.zeros(shape=3)
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
def unittorsion2(v1,v2,r3,bond_ang,tors_angle):
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
    rot_angle = np.arctan2(z1_s, y1_s)

#    print(rot_angle, tors_angle, rot_angle+tors_angle)
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

    v3 = np.zeros(shape=3)
    v3[0] = coeff1*v2[0] - coeff2*c_term*v2[1] - coeff3*s_term*v2[0]*v2[2]
    v3[1] = coeff1*v2[1] + coeff2*c_term*v2[0] - coeff3*s_term*v2[1]*v2[2]
    v3[2] = coeff1*v2[2]                       + coeff3*s_term*(r_proj*r_proj)


    return v3

#===============================================================
def unittorsion(v1, v2, r3, bond_ang, tors_angle):

    #Rebuilds a new atom from the positions of 3 other atoms
    #along with the bond distance, angle, and torsional angle
    #
    #v1 - v2 - v3

    v1_u = v1-v2

    w1 = np.copy(v2)
    w1 /= np.linalg.norm(w1)
#    print("w1",w1)

    w2 = np.array([w1[1], -w1[0], 0.0])
    w2 /= np.linalg.norm(w2)
#    print("w2",w2)

    w3 = np.cross(w1, w2)
    w3 /= np.linalg.norm(w3)

    vec_mat = np.stack([w1, w2, w3], axis=1).transpose()
    vec_mat_inv = np.linalg.inv(vec_mat)
#    print(vec_mat)
#    print(vec_mat_inv)
#    print("w3",w3)

#    print("check:",np.dot(w1,w2))
#    print("check:",np.dot(w1,w3))
#    print("check:",np.dot(w2,w3))

    y1_s = np.dot(v1_u, w2)
    z1_s = np.dot(v1_u, w3)

    v1_angle = np.arctan2(z1_s, y1_s)
    v1_angle = anglebounds(v1_angle)
#    print("Torsion", v1_angle, tors_angle, v1_angle+tors_angle)

    #Determine the new position we need to rotate the vector to in order to create
    #the proper torsional geometry.


#    print("bond ang",bond_ang)
    c_theta = np.cos(bond_ang)
    s_theta = np.sin(bond_ang)
#    print("bond ang cos", c_theta)
#    print("bond ang sin", s_theta)


    rotmat1 = np.array([[c_theta, s_theta, 0.0],[-s_theta, c_theta, 0.0], [0.0, 0.0, 1.0]])
#    print("rot1")
#    print(rotmat1)


    c = np.array([r3, 0.0, 0.0])
#    print("c:", c)
    c = np.matmul(rotmat1, c) #Rotate it along the x,y axis to create the bond angle
#    print(c)
#    print("Rot 1 Mag:", np.linalg.norm(c))

    #Figure out where the new vector is located in the y-z axis and figure out where we need move it to.
    cur_angle = np.arctan2(c[2], c[1]) 
    cur_angle = anglebounds(cur_angle)
#    print("v1_angle:",v1_angle)
#    print("cur_angle:",cur_angle)
    phi = -(v1_angle + tors_angle - cur_angle)
    phi = anglebounds(phi)

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

#    print("dihedral ang ", phi)
#    print("dihedral ang cos", c_phi)
#    print("dihedral ang sin", s_phi)
    rotmat2 = np.array([[1.0, 0.0, 0.0], [0.0, c_phi, s_phi],[0.0, -s_phi, c_phi] ])

#    print("rot2")
#    print(rotmat2)
    c = np.matmul(rotmat2, c) #Rotate it along the y,z axis to create the dihedral angle
    v3 = np.matmul(vec_mat_inv, c)
#    print(v3)
#    print(np.linalg.norm(v3))
#    print(np.arccos(np.dot(v3,v2)))

#    testang = np.arccos(np.dot(v3, v2) / (np.linalg.norm(v3) * np.linalg.norm(v2)))
#    print("post rot", testang)
#    print("post rot mag", np.linalg.norm(v3))

    v1_s = np.array([np.dot(v1_u, w2), np.dot(v1_u, w3)])
    v3_s = np.array([np.dot(v3, w2), np.dot(v3, w3)])

    testang = np.arccos(np.dot(v3_s, v1_s) / (np.linalg.norm(v3_s) * np.linalg.norm(v1_s)))
#    print("post phi", testang)


    y3_s = np.dot(v3, w2) 
    z3_s = np.dot(v3, w3)
    cur_angle = np.arctan2(z3_s, y3_s)
#    print(cur_angle, v1_angle)

#    print()
    return v3
#============================
def anglebounds(angle):
    while angle < 0.0:
       angle = angle + 2.0*np.pi
    while angle > 2.0*np.pi:
       angle = angle - 2.0*np.pi
    return angle
#=====================================================
