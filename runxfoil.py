# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:30:07 2021

@author: zheng
"""


def runxfoil(name):
    import numpy as np
    import os
    import time
    AoA        = '0'
    Re         = '5000000'
    xfoilFlnm  = 'xfoil_input.txt'
    fid = open(xfoilFlnm,"w")
    fid.write('PLOP\n')
    fid.write('G\n')
    fid.write('\n')
    fid.write("load \n")
    fid.write(name + "\n")
    fid.write('\n') 
    fid.write('PANE\n')
    fid.write('PPAR\n')
    fid.write('R\n')
    fid.write('0.2\n')
    fid.write('\n')
    fid.write('\n')
    fid.write('\n')
    fid.write('\n')
    fid.write("OPER\n")
    fid.write("V\n")
    fid.write(Re +'\n')
    fid.write('pacc\n')
    fid.write('save.txt\n')
    fid.write('\n') 
    fid.write("A\n" )
    fid.write(AoA + "\n")
    fid.close()
    os.system("xfoil.exe < xfoil_input.txt")
    time.sleep(0.05)
    marker = 0
    try:
        dataBuffer = np.loadtxt('save.txt',skiprows = 12)
        cl = dataBuffer[1]
        cd = dataBuffer[2]
        cm = dataBuffer[4]
        marker = 1
    except: 
        marker = 0
        cl = 0
        cd = 0
        cm = 0;
    time.sleep(0.05)
    if os.path.exists('save.txt'):
        try:
            os.remove('save.txt')
        except:
            pass
    return [cl,cd,cm,marker]