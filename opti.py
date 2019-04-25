# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:39:53 2019

@author: kst
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#import mpmath as p
from random import shuffle
lam1 =0
lam2 =0
alpha = 0.1
n =100
xinit = [0]
l = np.zeros(n)
l2 = np.zeros(n)
x1l = np.zeros(n)
x2l = np.zeros(n)
x3l = np.zeros(n)
t = np.linspace(0,5,100)

m = lambda x: (x[0] - 1)**2+(x[1]-5)**2 + (x[2]-2)**2

cons = [{'type': 'eq', 'fun': lambda x:  -x[0] + 2*x[1] - x[2] - 2},
        {'type': 'eq', 'fun': lambda x:  5*x[0] + 7*x[1] -3*x[2] - 5}
        ]

res1 = minimize(m, x0=[0,0,0], constraints=cons)

