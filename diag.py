#----------------------------------------------------#
#   密度矩阵属于厄米矩阵
#   厄米矩阵，不同特征值对应的特征向量正交
#   n阶方阵可对角化的条件是有n个线性无关的特征向量，因此密度矩阵对角化后的正交基就是它的特征向量
#----------------------------------------------------#
import numpy as np
from numpy import random,mat

# 返回一个列表，第一个元素为特征值，第二个元素为正交基
def Diag_orthbase(rho):
    # cval得到特征值和特征向量,一列为一个特征向量
    cval = np.linalg.eig(rho)
    eigenval = cval[0]
    basevec = mat(cval[1])
    # diag为对角化矩阵
    diag = mat(np.linalg.inv(cval[1])) *mat(rho) *mat(cval[1])
    # 验证是否对角化成功a 0><0 + b 1><1  
    p = eigenval[0] *mat(basevec[:,0].reshape(-1,1)) *mat(basevec[:,0].T) + eigenval[1] *mat(basevec[:,1].reshape(-1,1)) *mat(basevec[:,1].T)
    return cval

print(Diag_orthbase([[1/3,3],[3,2/3]]))

  