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
import tensorflow as tf
from tensorflow import keras
modelCD = keras.models.load_model("PredictCDby100")
modelCL = keras.models.load_model("PredictCLby1")
modelCM = keras.models.load_model("PredictCMby1")

def myfunc(x,grad):
    historyObj = open('historyCNN.txt','a')
    piter = x
    dcddxCNN = np.ones([1,16])[0]
    dcldxCNN = np.ones([1,16])[0]

    DATAX = []
    XX,YY = ffd_opt.ffd_opt(x)
    fig = plt.figure(figsize=(2,2),dpi=100)            
    plt.fill(XX,YY,'k')
    plt.axis('off')
    plt.ylim([-0.3,0.3])
    plt.savefig('iterate.png')
    plt.close(fig)
    image = Image.open('iterate.png')
    image = image.convert('L')
    image = image.resize((128,128))
    data = np.asarray(image)
    data2 = np.reshape(data,-1)
    DATAX = np.append(DATAX,data2)
    DATAX = [DATAX]
    for i in range(len(x)):
        piter = copy.copy(x)
        piter[i] = x[i]+0.02
        XX,YY = ffd_opt.ffd_opt(piter)
        fig2 = plt.figure(figsize=(2,2),dpi=100)            
        plt.fill(XX,YY,'k')
        plt.axis('off')
        plt.ylim([-0.3,0.3])
        plt.savefig('iterate.png')
        plt.close(fig2)
        image = Image.open('iterate.png')
        image = image.convert('L')
        image = image.resize((128,128))
        data = np.asarray(image)
        data2 = np.reshape(data,-1)
        DATAX = np.append(DATAX,[data2],axis = 0)
    X= DATAX.reshape(-1,128,128,1)
    cl = modelCL.predict(X)
    cd = (modelCD.predict(X))/100
    #Specify weight for cl and cd and compute objective
    w1 = 0.8;
    w2 = 0.02;
    clstar = 0.3;
    cdstar = 0.01;
    F = float(w1*(-cl[0]/clstar +1)**2+w2*(-cd[0]/cdstar +1)**2)
    print(str(F)+'\n')
    for i in range(1,len(cl)):
        dcldxCNN[i-1] = (cl[i]-cl[0])/(0.02)
        dcddxCNN[i-1] = (cd[i]-cd[0])/(0.02)
    if grad.size>0:
        grad[:] = (w1*2*(-cl[0]/clstar +1)*-1/clstar*dcldxCNN+\
            w2*2*(-cd[0]/cdstar +1)*-1/cdstar*dcddxCNN)*0.1
        print(grad)
        historyObj.write(str(F)+'\n')
    historyObj.close()
    return F

def myCons(x,grad,a):
    dcmdxCNN = np.ones([1,16])[0]
    DATAX = []
    piter = x
    XX,YY = ffd_opt.ffd_opt(x)
    fig = plt.figure(figsize=(2,2),dpi=100)            
    plt.fill(XX,YY,'k')
    plt.axis('off')
    plt.ylim([-0.3,0.3])
    plt.savefig('iterate.png')
    plt.close(fig)
    image = Image.open('iterate.png')
    image = image.convert('L')
    image = image.resize((128,128))
    data = np.asarray(image)
    data2 = np.reshape(data,-1)
    DATAX = np.append(DATAX,data2)
    DATAX = [DATAX]
    for i in range(len(x)):
        piter = copy.copy(x)
        piter[i] = x[i]+0.02
        XX,YY = ffd_opt.ffd_opt(piter)
        fig2 = plt.figure(figsize=(2,2),dpi=100)            
        plt.fill(XX,YY,'k')
        plt.axis('off')
        plt.ylim([-0.3,0.3])
        plt.savefig('iterate.png')
        plt.close(fig2)
        image = Image.open('iterate.png')
        image = image.convert('L')
        image = image.resize((128,128))
        data = np.asarray(image)
        data2 = np.reshape(data,-1)
        DATAX = np.append(DATAX,[data2],axis = 0)
    X= DATAX.reshape(-1,128,128,1)
    cm = modelCM.predict(X)
    M = float(cm[0]+a)
    for i in range(1,len(cm)):
        dcmdxCNN[i-1] = (cm[i]-cm[0])/(0.02)
    if grad.size>0:
        grad[:] = dcmdxCNN
    
    return M
    
import timeit
from numpy import loadtxt
try:
    os.remove('historyCNN.txt')
except:
    pass
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
opt.set_ftol_rel(1e-3)
opt.add_inequality_constraint(lambda x,grad: myCons(x,grad,0.05),1e-8)
x = opt.optimize(p0)
EndT = timeit.default_timer()
#%%
Runed = EndT-startT
#Plot the historical value of objective
lines = loadtxt('historyCNN.txt', comments="#", delimiter=" ", unpack=False)
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
fi = open('OptimizedCNN.txt','w')
for j in range(len(XX)):
    fi.write(str(XX[j])+ '    ' + str(YY[j])+'\n')
fi.close()
print('Time: ', Runed)  


