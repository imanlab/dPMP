import numpy as np
from numpy import sin, cos, sqrt
from scipy.spatial.transform import Rotation
"""Convert (roll, pitch, yaw) -> quaternion."""
def euler2quaternion(r, p, y):
    ci = cos(r/2.0)
    si = sin(r/2.0)
    cj = cos(p/2.0)
    sj = sin(p/2.0)
    ck = cos(y/2.0)
    sk = sin(y/2.0)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk
    return [cj*sc-sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]

"""
Calculate the rototranslation matrix.
Inputs are roll, pitch, yaw and (x, y, z) translation coordinates.
See: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
"""
def Trpyxyz(r, p, y, X, Y, Z):

    q = euler2quaternion(r, p, y)
    qi = q[0]
    qj = q[1]
    qk = q[2]
    qr = q[3]

    norm = sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
    qi = qi / norm
    qj = qj / norm
    qk = qk / norm
    qr = qr / norm

    T = np.array([
        [1 - 2 * (qj*qj + qk*qk), 2 * (qi*qj - qk*qr), 2 * (qi*qk + qj*qr), X],
        [2 * (qi*qj + qk*qr), 1 - 2 * (qi*qi + qk*qk), 2 * (qj*qk - qi*qr), Y],
        [2 * (qi*qk - qj*qr), 2 * (qj*qk + qi*qr), 1 - 2 * (qi*qi + qj*qj), Z],
        [0, 0, 0, 1]
    ])
    return T

"""
Calculate the rototranslation matrix for a pure rotation around the z axis.
See: https://en.wikipedia.org/wiki/Rotation_matrix
"""
def Trz(theta):

    T = np.array([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return T

"""
Compute the transformation matrix of the end-effector for a given configuration (MoveIt frame).
Values from URDF file https://github.com/StanfordASL/PandaRobot.jl/blob/master/deps/Panda/panda.urdf
"""
def FK(q):
    T01 = np.matmul(Trpyxyz(0, 0, 0, 0, 0, 0.333), Trz(q[0]))
    T12 = np.matmul(Trpyxyz(-1.57079632679, 0, 0, 0, 0, 0), Trz(q[1]))
    T23 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0, -0.316, 0), Trz(q[2]))
    T34 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0.0825, 0, 0), Trz(q[3]))
    T45 = np.matmul(Trpyxyz(-1.57079632679, 0, 0, -0.0825, 0.384, 0), Trz(q[4]))
    T56 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0, 0, 0), Trz(q[5]))
    T67 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0.088, 0, 0), Trz(q[6]))
    T78 = Trpyxyz(0, 0, 0, 0, 0, 0.107)
    T8B = Trpyxyz(0 ,-1.57, -3.925, 0, 0, 0)
    TBee = Trpyxyz( 0, 0, 0,0.25 ,0 ,-0.04)
    T08 = np.matmul(T01, np.matmul(T12, np.matmul(T23, np.matmul(T34, np.matmul(T45, np.matmul(T56, np.matmul(T67, np.matmul(T78, np.matmul(T8B, TBee)))))))))
    return T08

def joint_to_cartesian_ee_position(q):
    ee = FK(q)  # Transformation matrix.
    x = ee[0, 3]
    y = ee[1, 3]
    z = ee[2, 3]
    rot_matrix = ee[0:3, 0:3]
    r = Rotation.from_matrix(rot_matrix).as_quat()
    q1,q2,q3,q4 = r[0],r[1],r[2],r[3]
    return x,y,z,q1,q2,q3,q4

if __name__ == '__main__':
    home = [0.004046583856622568, -0.21545873952713104, -0.005451456256474691, -1.9326629693958524, -0.007122770030465392, 3.14, 0.8168725628418113]
    x,y,z,q1,q2,q3,q4 = joint_to_cartesian_ee_position(home)
    print(y)