# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 20:49:50 2022

@author: zheng
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import qmc
from io import StringIO
import os
sampler = qmc.LatinHypercube(d=16)
sampleN = 10                            #Of course this number will be 
                                    # larger when doing actual sampling
sample = sampler.random(n = sampleN)*0.04-0.02

# A FEW PARAMETERS FOR FLOW CONDITION
CFG = open('RunCFD.cfg','r') # this is the config file for direct analysis 

cfg_L = CFG.readlines()
mach = 0.2                        # change mach
AoA  = 0                          # change Angle of attack
Re = '6.5E6'                      # change Reynolds number
cfg_L[30] = 'MACH_NUMBER= '+str(mach) +'\n'   
cfg_L[32] = 'AOA= '+str(AoA) + '\n'
cfg_L[48] = 'REYNOLDS_NUMBER= '+Re +'\n'
CFG = open('RunCFD.cfg','w')
CFG.writelines(cfg_L) 
CFG.close()
#NOW THE DIRECT ANALYSIS CFD CONFIG IS SET
for i in range(sampleN):
###DEFORM MESH
    CFG = open('DefNACA0012.cfg','r') 
    cfg_L = CFG.readlines()
    dv = np.concatenate(([0], sample[i,0:8], [0], [0], sample[i,8:16], [0]))
    Sdv = ' '.join(str(elem) for elem in dv)
    cfg_L[370] = 'MESH_OUT_FILENAME= ' + str(i)+'_Def.su2 \n'
    cfg_L[467] = 'DV_VALUE=' +Sdv +'\n'
    CFG = open('DefNACA0012.cfg','w')
    CFG.writelines(cfg_L) 
    CFG.close()
    os.system('mpirun -n 12 SU2_DEF DefNACA0012.cfg')      #uncomment these two lines on a linux machine to call SU2

###RUN ANALYSIS
    CFG = open('RunCFD.cfg','r') 
    cfg_L[364] = 'MESH_FILENAME = '+str(i)+'_Def.su2 \n'
    cgf_L[400] = 'SURFACE_FILENAME= '+str(i)+'_surface_flow\n'
    cfg_L[382] = 'CONV_FILENAME= '+str(i)+'history\n'
    CFG = open('RunCFD.cfg','w')
    CFG.writelines(cfg_L) 
    CFG.close()
    os.system('mpirun -n 12 SU2_CFD RunCFD.cfg')

#### NOW WE HAVE A BUNCH OF .SU2 FILES, AND A BUNCH OF CONVERGENCE HISTORY FILES
#### WHICH INCLUDES CL CD DATA REQUIRED,
#### THE VTU FILES COULD BE USED TO VISUALIZE AIRFOILS
