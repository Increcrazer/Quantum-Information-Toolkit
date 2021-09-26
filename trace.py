#----------------------------------------------------#
#   两个量子态之间迹距离和保真度
#----------------------------------------------------#
from math import sqrt, atan, acos
import numpy as np
# sqrtm是对矩阵开根号，不是对矩阵元素开根号
from scipy.linalg import sqrtm
from sympy import symbols, solve, linsolve
from numpy import random, mat
from diag import Diag_orthbase
from bloch import Create_sphere, Bloch
from matplotlib import pyplot as plt 
from plottool import set_axes_equal

pi = np.pi
Iden = np.array([[1,0],[0,1]])
PauliX = np.array([[0,1],[1,0]])
PauliY = np.array([[0,complex(0,-1)],[complex(0,1),0]])
PauliZ = np.array([[1,0],[0,-1]])

# 通过定义式求迹距离
def Trace_dis(rho, sigma):
    # 求迹范数
    norm = sqrtm(np.dot(np.conj(rho - sigma).T, rho - sigma))
    return 0.5 *np.trace(norm)
    
# 通过Bloch球求迹距离
def Trace_dis_Bloch(rho, sigma):
    Dvec = Get_Bloch_vector_carte(rho) - Get_Bloch_vector_carte(sigma)
    return 0.5 *np.linalg.norm(Dvec) 

# 通过密度矩阵求在Bloch球上的向量(笛卡尔坐标系)
def Get_Bloch_vector_carte(rho):
    x, y, z = symbols('x y z')
    func_group = x *PauliX + y *PauliY + z *PauliZ + Iden - 2 *rho
    f1 = func_group[0,0]
    f2 = func_group[0,1]
    f3 = func_group[1,0]
    f4 = func_group[0,1]
    zxyval = np.array(list(solve([f1, f2, f3, f4]).values()), dtype=float)
    xyzval = zxyval[np.array([1,2,0])]
    return xyzval

# 通过密度矩阵求在Bloch球上的向量(球坐标系)
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

    theta = acos(z/r)
    return np.array([r,theta,phi])


# 通过定义式求保真度
def Fidelity(rho, sigma):
    return np.trace(sqrtm(mat(sqrtm(rho)) *sigma *sqrtm(rho)))

# 通过纯态和密度矩阵求保真度,它们的基底必须相同
def Fidelity_state_denma(phi,rho):
    return sqrtm(mat(phi) *rho *phi.reshape(-1,1))[0][0]
    

#----------------------------------------------------#
#   使用范例
# a = np.array([[1,1],[1,0]])
# b = np.array([[1/4,3],[3,3/4]])
# c = np.array([3/7,4/7])
# print(Get_Bloch_vector_carte(a))
# print(Get_Bloch_vector_carte(b))
# print(Get_Bloch_vector_sphere(a))

# fig = plt.figure()
# ax = Create_sphere(fig)
# Bloch(*Get_Bloch_vector_sphere(b),ax,'b')
# Bloch(*Get_Bloch_vector_sphere(a),ax,'a')
# set_axes_equal(ax)
# print(Trace_dis(a,b))
# print(Trace_dis_Bloch(a,b))
# print(Fidelity(a,b))
# print(Fidelity_state_denma(c,a))
# plt.show()
#----------------------------------------------------#




