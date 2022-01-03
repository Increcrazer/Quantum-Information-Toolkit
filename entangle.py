#----------------------------------------------------#
#   Entangle judgement
# * Copyright (C) 2021 Increcrazer
#----------------------------------------------------#
import numpy as np

# Positive partial transpositions, da *db, return matrix after PPT
def PPT(rho, da, db):
    rho = np.array(rho)
    rho_len = np.shape(rho)[0]
    step = int(rho_len/da)
    m = [[0] *da for row in range(da)]

    if da *db != rho_len:
        return 'Dimension not match'

    for i in range(da):
        for j in range(da):
            m[j][i] = rho[i*step:(i+1)*step,j*step:(j+1)*step].tolist()  # m is reversed in space a
    # print(np.transpose(m[0][1]).reshape(1,-1))     

    m_row = [[0] for row in range(da)]
    for i in range(0,da): 
        m_col = m[i][0] 
        for j in range(1,da):  
            m_col = np.append(m_col,m[i][j],axis=1)
        m_row[i] = m_col

    m_re = m_row[0]     # m_re is the matrix after PPT
    for i in range(1,da):
        m_re = np.append(m_re,m_row[i],axis=0)
    
    D = np.linalg.eigvals(m_re)      

    if da *db <= 6:
        if np.any(D < 0):    # if exist negative eigenvalue
            print('entangled')
        else:
            print('separable')
    else:
        if np.any(D < 0):    # if exist negative eigenvalue
            print('entangled')
        else:
            print('undetermined')
    
    return m_re


# Majorization, da *db
def Majorization(rho, rho_a, rho_b):
    da = np.shape(rho_a)[0]
    db = np.shape(rho_b)[0]
    if da *db != np.shape(rho)[0] or da != np.shape(rho_a)[0] or da != np.shape(rho_b)[0]:
        return 'Dimension not match'
    
    eigenval_rho = sorted(np.linalg.eig(rho)[0], reverse = True)     # sort from small to big，reverse invert the sequence
    eigenval_rho_a = sorted(np.linalg.eig(rho_a)[0], reverse = True)
    eigenval_rho_b = sorted(np.linalg.eig(rho_b)[0], reverse = True)

    eigenval_rho = list(eigenval_rho) + [0] *(da *db - len(eigenval_rho))
    eigenval_rho_a = list(eigenval_rho_a) + [0] *(da *db - len(eigenval_rho_a))
    eigenval_rho_b = list(eigenval_rho_b) + [0] *(da *db - len(eigenval_rho_b))

    for i in range(1,da *db):    # i:1~4
        if sum(eigenval_rho[0:i]) <= sum(eigenval_rho_a[0:i]) and sum(eigenval_rho[0:i]) <= sum(eigenval_rho_b[0:i]):
            continue
        else:
            break
    print('undetermined') if i == 4 else print('entangled')

# Reduction, da *db
def Reduction(rho, rho_a):
    rho_a_len = np.shape(rho_a)[0]
    eigenval = np.linalg.eigvals(np.kron(rho_a,np.identity(rho_a_len)) - rho)

    if np.shape(rho)[0] == 4:
        if np.any(eigenval < 0):    # if exist negative eigenvalue
            print('distillable')
        else:
            print('separable')
    else:
        if np.any(eigenval < 0):
            print('distillable')
        else:
            print('undetermined')


# Realignment, da *db, strong enough to distinguish most of BES 
def Realignment(rho, da, db):
    k = 0
    rho = np.array(rho)
    rho_len = np.shape(rho)[0]
    step = int(rho_len/da)
    m = [[0] * da for row in range(da)]
    m_re = [[0] * db **2 for row in range(da **2)]

    if da *db != rho_len:
        print('Dimension not match')

    for i in range(da):
        for j in range(da):
            m[j][i] = rho[i*step:(i+1)*step,j*step:(j+1)*step]  # m is the element of the matrix partitioned, and transposed in space a
    # print(np.transpose(m[0][1]).reshape(1,-1))     
    for i in range(da):
        for j in range(da):
            m_re[k] = np.transpose(m[i][j]).reshape(1,-1)[0].tolist()
            k = k + 1

    if sum(np.sqrt(np.linalg.eigvals(np.dot(m_re,np.conj(m_re).T)))) > 1:
        print('entangled')
    else:
        print('undetermined')
                
       
# usage
# ----------------------------------------------------#
# m = [[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]] 
# m1 = [[0.5,0],[0,0.5]]
# m2 = [[0.5,0],[0,0.5]]
# print(PPT(m,2,2))   
# Majorization(m,m1,m2)
# Reduction(m,m1)
# Realignment(m,2,2)
# ----------------------------------------------------#



