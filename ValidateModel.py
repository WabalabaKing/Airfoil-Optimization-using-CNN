# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:30:23 2021

@author: zheng
"""


import os
import tensorflow as tf
from tensorflow import keras
import scipy.io
import numpy as np
import PredictCD
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
dataT = scipy.io.loadmat('Test.mat')
Xt,yt = dataT['datax'],dataT['datay']
X_test = Xt.reshape(-1,128,128,1)
true = np.reshape(yt[:,2],(len(yt[:,1]),-1))
cl_predict, cd_predict,predict = PredictCD.PredictCD(X_test)
#%%
plt.figure(figsize=(5,5))
plt.scatter(true, predict,s=1)
plt.plot([-2, 1], [-2, 1], 'k--', lw=1)
#plt.plot([-3*scl, -3*scl+1], [0, 1], '--', lw=1)
#plt.plot([3*scl, 3*scl+1], [0, +1], '--', lw=1)
plt.ylim([-0.5, 0.25])
plt.xlim([-0.5, 0.25])
res = r2_score(true,predict)
plt.xlabel('Predicted Cm Value')
plt.ylabel('Actual Cm Value')
plt.title(' Prediction of Moment R2 Score=' + str(res))
plt.show()
