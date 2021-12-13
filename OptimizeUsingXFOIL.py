# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:23:38 2021

@author: zheng
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:21:36 2021

@author: zheng
"""


import nlopt
from numpy import *
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
from PIL import Image
import scipy.io as sio
import PredictCD
import copy
import runxfoil
import ffd_opt

def myfunc(p,grad):
    historyObj = open('historyxfoil.txt','a')
    dcddxFD = np.ones(16)
    dcldxFD = np.ones(16)
    XX,YY = ffd_opt.ffd_opt(p)
    fi = open('base.txt','w')
    for j in range(len(XX)):
        fi.write(str(XX[j])+ '    ' + str(YY[j])+'\n')
    fi.close()
    clb,cdb,cm,m = runxfoil.runxfoil('base.txt')
    for i in range(len(p)):
        piter = copy.copy(p)
        piter[i] = p[i]+1e-3
        XX,YY = ffd_opt.ffd_opt(piter)
        fi = open('iterate.txt','w')
        for j in range(len(XX)):
            fi.write(str(XX[j])+ '    ' + str(YY[j])+'\n')
        fi.close()
        clx,cdx,cm,m = runxfoil.runxfoil('iterate.txt')
        dcldxFD[i] =(clx-clb)/(1e-3)
        dcddxFD[i] =(cdx-cdb)/(1e-3)
    #Specify weight for cl and cd and compute objective
    w1 = 0.8;
    w2 = 0.02;
    clstar = 0.3;
    cdstar = 0.01;
    F = float(w1*(-clb/clstar +1)**2+w2*(-cdb/cdstar +1)**2)
    print(str(F)+'\n')
    if grad.size>0:
        grad[:] = (w1*2*(-clb/clstar +1)*-1/clstar*dcldxFD+\
            w2*2*(-cdb/cdstar +1)*-1/cdstar*dcddxFD)*0.1
        print(grad)
        historyObj.write(str(F)+'\n')
    historyObj.close()
    return F

def myCons(p,grad,a):
    dcmdxFD = np.ones(16)
    XX,YY = ffd_opt.ffd_opt(p)
    fi = open('base.txt','w')
    for j in range(len(XX)):
        fi.write(str(XX[j])+ '    ' + str(YY[j])+'\n')
    fi.close()
    clb,cdb,cmb,m = runxfoil.runxfoil('base.txt')
    for i in range(len(p)):
        piter = copy.copy(p)
        piter[i] = p[i]+1e-3
        XX,YY = ffd_opt.ffd_opt(piter)
        fi = open('iterate.txt','w')
        for j in range(len(XX)):
            fi.write(str(XX[j])+ '    ' + str(YY[j])+'\n')
        fi.close()
        clx,cdx,cmx,m = runxfoil.runxfoil('iterate.txt')
        dcmdxFD[i] =(cmx-cmb)/(1e-3)
    #Specify weight for cl and cd and compute objective

    M = cmb+a
    if grad.size>0:
        grad[:] = dcmdxFD
    return M

import timeit
try:
    os.remove('historyxfoil.txt')
except:
    pass
from numpy import loadtxt
startT = timeit.default_timer()
pu =np.append(0.2*np.ones([1,8]),0*np.ones([1,8]))
pd =np.append(0*np.ones([1,8]),-0.12*np.ones([1,8]))
p0 = np.append(0.1*np.ones([1,8]),-0.1*np.ones([1,8]))
dp = 0.01*np.ones(16)
opt = nlopt.opt(nlopt.LD_MMA, 16)
opt.set_min_objective(myfunc)
opt.set_lower_bounds(pd)
opt.set_upper_bounds(pu)
opt.set_initial_step(dp)
opt.add_inequality_constraint(lambda x,grad: myCons(x,grad,0.05),1e-8)
opt.set_ftol_rel(1e-3)
x = opt.optimize(p0)
EndT = timeit.default_timer()
#%%
Runed = EndT-startT
#Plot the historical value of objective
lines = loadtxt('historyxfoil.txt', comments="#", delimiter=" ", unpack=False)
var = np.linspace(1,lines.size,num=lines.size)
plt.plot(var,lines)
plt.xlabel('Iteration')
plt.ylabel('Objective Function')
plt.title('Optimization History, RunTime = '+str(Runed))


XX,YY = ffd_opt.ffd_opt(x)
DATAX = []
plt.figure(figsize=(2,2),dpi=100)            
plt.fill(XX,YY,'k')
plt.axis('on')
plt.axis('equal')
plt.savefig('iterate.png')
plt.show()
image = Image.open('iterate.png')
image = image.convert('L')
image = image.resize((128,128))
data = np.asarray(image)
data2 = np.reshape(data,-1)
DATAX = np.append(DATAX,data2)
DATAX = [DATAX]
DATAX = np.append(DATAX,[data2],axis = 0)
cl,cd = PredictCD.PredictCD(DATAX)
fi = open('OptimizedXFOIL.txt','w')
for j in range(len(XX)):
    fi.write(str(XX[j])+ '    ' + str(YY[j])+'\n')
fi.close()
print('Time: ', Runed)  


