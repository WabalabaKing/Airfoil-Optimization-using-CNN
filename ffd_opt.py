# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 23:55:05 2021

@author: zheng
"""


def ffd_opt(p):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import comb
    xu = np.append(p[0:8],0.1)
    xl = np.append(p[8:16],-0.1)
    xu = np.append(0.1,xu)
    xl = np.append(-0.1,xl)
    n = 10
    p_up = np.array([0.00498078078183743,0.11369873675813,-0.0760156974911251,0.321157718313777,-0.284633644566237,0.399891607417138,-0.199316699573229,0.162265419654904,-0.0180098813758781,0.0235995110841613,7.27429161513346e-05])
    p_down = np.array([-0.00497989295236084,-0.113736339850011,0.0757975441991346,-0.321004761990008,0.284609647790527,-0.399650797760633,0.199186822010331,-0.162384831897831,0.0178462941230894,-0.0235970228211667,-7.29204546219269e-05])
    pupM =0.1-p_up
    pdnM = -0.1-p_down
    U = xu
    D = xl
    t = np.linspace(0,np.pi,num=90)
    X = 0.5+np.flip(np.cos(t)/2)
    UPFFD = (1-X)**n*(U[0]-pupM[0])+X**n*(U[-1]-pupM[-1]);
    DNFFD = (1-X)**n*(D[0]-pdnM[0])+X**n*(D[-1]-pdnM[-1]);
    for j in range(1,n):
        UPFFD = UPFFD+comb(n,j)*(1-X)**(n-j)*X**j*(U[j]-pupM[j]);
        DNFFD = DNFFD+comb(n,j)*(1-X)**(n-j)*X**j*(D[j]-pdnM[j]);
    XX = np.append(np.flip(X)[0:-1],X)
    YY = np.append(np.flip(UPFFD)[0:-1],DNFFD)
    return XX,YY
