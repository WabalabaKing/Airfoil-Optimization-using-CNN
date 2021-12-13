# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 21:40:36 2021

@author: zheng
"""


import os
import matplotlib
import numpy as np
import scipy
from PIL import Image
import scipy.io as sio
DATAX = np.array([])
DATAY = np.array([])
mark = 0
for filename in os.listdir('Airfoil_Image_Valid'):
    dirPname = 'Airfoil_Image_Valid/'+filename
    name = filename[0:-4] 
    coefname = 'Airfoil_Coeff_Valid/'+ name+'.dat'
    coef = np.genfromtxt(coefname)
    cl = coef[0]
    cd = coef[1]
    cm = coef[2]
    image = Image.open(dirPname)
    image = image.convert('L')
    image = image.resize((128,128))
    data = np.asarray(image)
    data2 = np.reshape(data,-1)
    if mark ==0:
        DATAX = np.append(DATAX,data2)
        DATAY = np.append(DATAY,[cl,cd,cm])
        DATAX = [DATAX]
        DATAY = [DATAY]
        mark = 1
    else:
        DATAX = np.append(DATAX,[data2],axis = 0)
        DATAY = np.append(DATAY,[[cl,cd,cm]],axis = 0)

adict = {}
adict['datax'] = DATAX
adict['datay'] = DATAY
sio.savemat('Valid.mat',adict)