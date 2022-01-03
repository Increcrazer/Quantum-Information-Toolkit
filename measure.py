#----------------------------------------------------#
#   Eigenvalue, measure, partial trace, and Schimidt decomposition
# * Copyright (C) 2021 Increcrazer
#----------------------------------------------------#
import numpy as np
from math import sqrt
from numpy import mat

# Get eigenvalue and according basevector
#----------------------------------------------------#
def Diag_orthbase(rho):
    # cval get eigenvalue and base, the base is set in colume, thus should be transversed
    cval = np.linalg.eig(rho)
    eigenval = cval[0]
    basevec = mat(cval[1]).T
    print('eigenvalue:\n',eigenval, '\nbasevec:\n',basevec)
#----------------------------------------------------#


# Measure pure state, see Nielson pp.85 and pp.88
#----------------------------------------------------#
# probability that m occurs
def Measure_pure_pro(M_m, phi_ket):
    KET = np.dot(M_m, phi_ket)
    BRA = np.conjugate(KET).T
    return np.dot(BRA,KET)

# the state after the measurement
def Measure_pure_state(M_m, phi_ket):
    return np.dot(M_m, phi_ket) /sqrt(Measure_pure_pro(M_m, phi_ket))

# the average value of the observable M 
def Measure_pure_ave(M, phi_ket):
    phi_bra = np.conjugate(phi_ket).T
    return phi_bra @M @phi_ket
#----------------------------------------------------#


# Measure mixed state, see Nielson pp.99
#----------------------------------------------------#
# probability that m occurs
def Measure_mix_pro(M_m, rho):
    return np.trace(np.conjugate(M_m).T @M_m @rho)

# the state after the measurement
def Measure_mix_state(M_m, rho):
    return np.array(M_m) @rho @ np.conjugate(M_m).T /Measure_mix_pro(M_m, rho)

# the average value of the observable Omega 
def Measure_mix_ave(rho, Omega):
    return np.trace(np.dot(rho, Omega))
#----------------------------------------------------#


# Partial trace, da *db
#----------------------------------------------------#
def Partial_trace(rho, da, db, par):
    rho = np.array(rho)
    rho_len = np.shape(rho)[0]
    step = int(rho_len/da)
    if da *db != rho_len:
        return 'Dimension not match'

    m = [[0] *da for row in range(da)]
    for i in range(da):
        for j in range(da):
            m[i][j] = rho[i*step:(i+1)*step,j*step:(j+1)*step].tolist()  # m is reversed in space a

    # partial trace on space a and get rho_B
    if par == 'A':
        rho_part = np.array(m[0][0])
        for i in range(1,da):
            rho_part = rho_part + m[i][i]
        return rho_part

    # partial trace on space b and get rho_A
    elif par == 'B':
        rho_part = [[0] *da for row in range(da)]
        for i in range(da):
            for j in range(da): 
                rho_part[i][j] = np.trace(m[i][j]) 
        return np.array(rho_part)
      
    else:
        print('please input A or B')
#----------------------------------------------------#


# Schimidt decomposition, da must = db, 0/1 base
# par is A_ij, where aij is coefficient before i>j>, see Nielson pp.109
# output i_A, i_B as row vector 
#----------------------------------------------------#
def Schmidt(par):
    u,v,w=np.linalg.svd(par)
    v=np.diag(v)
    i_A =  u.T
    i_B =  w
    print('lamda:\n',v, '\ni_A:\n',i_A, '\ni_B:\n',i_B)
#----------------------------------------------------#


# usage
#----------------------------------------------------#
# M0 = [[1,0],[0,0]]
# phi = [sqrt(3)/7,2/7]
# rho = [[1/5,3],[3,4/5]]
# d = [[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]]
# e = [[1/(2 *sqrt(2)),sqrt(3)/(2 *sqrt(2))],[sqrt(3)/(2 *sqrt(2)),1/(2 *sqrt(2))]]

# Diag_orthbase(e)

# print(Measure_pure_pro(M0,phi))
# print(Measure_pure_state(M0,phi))
# print(Measure_pure_ave(M0,phi))

# print(Measure_mix_pro(M0,rho))
# print(Measure_mix_state(M0,rho))
# print(Measure_mix_ave(rho,M0))

# print(Partial_trace(d,2,2,'B'))

# Schmidt(e)
#----------------------------------------------------#



  