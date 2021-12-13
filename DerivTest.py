# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 23:15:54 2021

@author: zheng
"""
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
import parsec
p=[0.0146,0.3025,0.06,-0.4928,0.3016,0.06,-0.4848,-0.0039,0.0101,-2.7791,9.2496]
#p = np.append(0.1*np.ones([1,9]),-0.1*np.ones([1,9]))

################################### TEST MODEL SENSITIVITY#######################
piter = p
dcddxCNN = np.ones([1,11])[0]
dcldxCNN = np.ones([1,11])[0]
dcddxFD = np.ones([1,11])[0]
dcldxFD = np.ones([1,11])[0]
DATAX = []
XX,YY = parsec.parsec(p)
plt.figure(figsize=(2,2),dpi=100)            
plt.fill(XX,YY,'k')
plt.axis('off')
plt.ylim([-0.3,0.3])
plt.savefig('iterate.png')
plt.show()
image = Image.open('iterate.png')
image = image.convert('L')
image = image.resize((128,128))
data = np.asarray(image)
data2 = np.reshape(data,-1)
DATAX = np.append(DATAX,data2)
DATAX = [DATAX]
for i in range(len(p)):
    piter = copy.copy(p)
    piter[i] = p[i]+0.001
    XX,YY = parsec.parsec(piter)
    plt.figure(figsize=(2,2),dpi=100)            
    plt.fill(XX,YY,'k')
    plt.axis('off')
    plt.ylim([-0.3,0.3])
    plt.savefig('iterate.png')
    plt.show()
    image = Image.open('iterate.png')
    image = image.convert('L')
    image = image.resize((128,128))
    data = np.asarray(image)
    data2 = np.reshape(data,-1)
    os.remove('iterate.png')
    DATAX = np.append(DATAX,[data2],axis = 0)
cl,cd,cm = PredictCD.PredictCD(DATAX)
for i in range(1,len(cl)):
    dcldxCNN[i-1] = (cl[i]-cl[0])/(0.001)
    dcddxCNN[i-1] = (cd[i]-cd[0])/(0.001)
################################### TEST MODEL SENSITIVITY#######################
XX,YY = parsec.parsec(p)
fi = open('base.txt','w')
for j in range(len(XX)):
    fi.write(str(XX[j])+ '    ' + str(YY[j])+'\n')
fi.close()
clb,cdb,cmb,m = runxfoil.runxfoil('base.txt')
print(str(m))
for i in range(len(p)):
    piter = copy.copy(p)
    piter[i] = p[i]+0.01
    XX,YY = parsec.parsec(piter)
    fi = open('iterate.txt','w')
    for j in range(len(XX)):
        fi.write(str(XX[j])+ '    ' + str(YY[j])+'\n')
    fi.close()
    clx,cdx,cmb,m = runxfoil.runxfoil('iterate.txt')
    print(str(m)+'  '+str(clx))
    dcldxFD[i] =(clx-clb)/(1e-2)
    dcddxFD[i] =(cdx-cdb)/(1e-2)
#%%    
var = np.linspace(1,11,num=11)
plt.plot(var,dcldxCNN, label='CNN')
plt.plot(var,dcldxFD, label='Potential Solver')
plt.legend()
plt.xlabel('Design Point')
plt.ylabel('Derivative of Cl')
plt.title('Sensitivity of Lift on NACA 0012 with FFD Parametrization')

plt.show()