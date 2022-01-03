#----------------------------------------------------#
#   Bloch vector, trace distance, fidelity, and Von entropy 
# * Copyright (C) 2021 Increcrazer
#----------------------------------------------------#

from math import sqrt, atan, acos
import numpy as np
# sqrtm is for matrix
from scipy.linalg import sqrtm, logm
from math import log,log2, sqrt
from sympy import symbols, solve
from numpy import  mat

# Define parameter
pi = np.pi
Iden = np.array([[1,0],[0,1]])
PauliX = np.array([[0,1],[1,0]])
PauliY = np.array([[0,complex(0,-1)],[complex(0,1),0]])
PauliZ = np.array([[1,0],[0,-1]])

# Tool func
#----------------------------------------------------#
# ita, used in Shannon entropy
def ita(x):
    return - x *log2(x)

# log of matrix，warning: use logm will set 0 as -inf???
def log2m(m):
    x = logm(m)/log(2)
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    return x
#----------------------------------------------------#


# Get Bloch vector
#----------------------------------------------------#
# Get Bloch vector by density matrix(cartesian)
def Get_Bloch_vector_carte(rho):
    x, y, z = symbols('x y z')
    func_group = x *PauliX + y *PauliY + z *PauliZ + Iden - 2 *np.array(rho)
    f1 = func_group[0,0]
    f2 = func_group[0,1]
    f3 = func_group[1,0]
    f4 = func_group[0,1]
    zxyval = np.array(list(solve([f1, f2, f3, f4]).values()), dtype=float)
    xyzval = zxyval[np.array([1,2,0])]
    return xyzval

# Get Bloch vector by density matrix(spherical)
def Get_Bloch_vector_sphere(rho):
    x = Get_Bloch_vector_carte(rho)[0]
    y = Get_Bloch_vector_carte(rho)[1]
    z = Get_Bloch_vector_carte(rho)[2]
    r = sqrt(x**2 + y**2 +z**2)
    if x*y != 0:
        phi = atan(y/x) if y>0 else atan(y/x) + pi
    elif x == 0 and y > 0:
        phi = pi/2
    elif x == 0 and y < 0:
        phi = 3*pi/2
    elif x > 0 and y == 0:
        phi = 0
    elif x <= 0 and y == 0: 
        phi = pi   

    if r != 0:
        theta = acos(z/r)
    else: 
        theta = 0

    return np.array([r,theta,phi])
#----------------------------------------------------#


# Trace distance
#----------------------------------------------------#
# Get trace distance by two density matrix through defination
def Trace_dis(rho, sigma):
    rho = np.array(rho)
    sigma = np.array(sigma)
    # trace norm
    norm = sqrtm(np.dot(np.conj(rho - sigma).T, rho - sigma))
    return 0.5 *np.trace(norm)
    
# Get trace distance by two density matrix through Bloch vector
def Trace_dis_Bloch(rho, sigma):
    Dvec = Get_Bloch_vector_carte(rho) - Get_Bloch_vector_carte(sigma)
    return 0.5 *np.linalg.norm(Dvec) 
#----------------------------------------------------#


# Fidelity
#----------------------------------------------------#
# Get fidelity by density matrix through defination
def Fidelity(rho, sigma):
    return np.trace(sqrtm(mat(sqrtm(rho)) *sigma *sqrtm(rho)))

# Get fidelity by pure state phi and density matrix rho
def Fidelity_state_denma(phi,rho):
    phi = np.array(phi)
    return sqrtm(mat(phi) *rho *phi.reshape(-1,1))[0][0]
#----------------------------------------------------#


# VEntropy
#----------------------------------------------------#
# Get VEntropy by density matrix in
def VEntropy(rho):
    return -np.trace(np.dot(rho, log2m(rho)))

# Relevant VEntropy
def RvEntropy(rho,sigma):
    return np.trace(np.dot(rho, log2m(rho))) - np.trace(np.dot(rho, log2m(sigma)))

# Fannes bound: upper bound of VEntropy between two density matrix
def Fannes(rho,sigma):
    return 2 *Trace_dis(rho,sigma) *log2(rho.shape[1]) + ita(2 *Trace_dis(rho,sigma))
#----------------------------------------------------#


# usage
#----------------------------------------------------#
# rho1 = [[0.5,0],[0,0.5]]
# rho2 = [[1,3],[3,0]]
# phi = [sqrt(3)/7,2/7]
# U = np.array([[sqrt(3)/2,-1/2], [1/2,sqrt(3)/2]])

# print(Get_Bloch_vector_carte(rho1))
# print(Get_Bloch_vector_sphere(rho1))

# print(Trace_dis(rho1,rho2))
# print(Trace_dis_Bloch(rho1,rho2))

# print(Fidelity(rho1,rho2))
# print(Fidelity_state_denma(phi,rho2))

# print(VEntropy(rho1)) 
# print(VEntropy(U @rho1 @U.T))
# print(RvEntropy(rho1,rho2))

# fig = plt.figure()
# ax = Create_sphere(fig)
# Bloch(*Get_Bloch_vector_sphere(rho1),ax,'a')
# set_axes_equal(ax)
# plt.show()
#----------------------------------------------------#




