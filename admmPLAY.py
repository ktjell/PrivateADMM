# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:35:19 2019

@author: kst
"""


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import mpmath as p
from random import shuffle
lam =0
alpha = 0.1
n = 100
xinit = [0]
l = np.zeros(n)
x1l = np.zeros(n)
x2l = np.zeros(n)
x3l = np.zeros(n)
t = np.linspace(0,5,100)

#ft = lambda x2,x3: 2*x2 -x3 -2
#tx = np.linspace(-5,5,100)
#ty = np.linspace(-5,5,100)
#
#X,Y = np.meshgrid(tx,ty)
#
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X,Y,ft(X,Y), 100)
#fig.show()
signG = np.zeros(n)
x2 = 0
x1=0
x3 = 0
rho = 5
for i in range(n):
    rho1 = np.random.uniform(1,2)
    rho2 = np.random.uniform(1,5)
    rho3 = np.random.uniform(1,5)
#    rho3 = rho1
#    rho2 = rho1
    sq = 4*1/2*rho1
    c = [-lam, 4*1/2*rho1,4*1/2*rho1*x2, 1/2*rho1*2*x3]
    shuffle(c)
    c.append(sq)
    c1 = c
    
#    c1 = [4* 1/2*rho2, 1/2*rho2*8,1/2*4*rho2*x1,1/2*rho2*4*x3 ,2*lam ]
    plt.plot(0,c1[0], 'bo')
    plt.plot(1,c1[1], 'ro')
    plt.plot(2,c1[2], 'go')
    plt.plot(3,c1[3], 'yo')
    plt.plot(4,c1[4], 'mo')
    o1 = lambda x: (x-1)**2  + (1/2*rho1)*x*x  + 4*1/2*rho1*x - 4*1/2*rho1*x2*x + 1/2*rho1*2*x3*x -lam*x
    
    o2 = lambda x: (x-5)**2  + 4* 1/2*rho2*x*x - 1/2*rho2*8*x - 1/2*4*rho2*x1*x - 1/2*rho2*4*x3*x + 2*lam*x
    
    o3 = lambda x: (x-2)**2  + 1/2*rho3*x*x + 1/2*rho3*4*x + 2*1/2*rho3*x1*x - 4*1/2*rho3*x2*x  -lam*x
    
    res1 = minimize(o1, x0=xinit)#, bounds=bnds)#, constraints=cons)
    x1 = res1['x'][0]
    res2 = minimize(o2, x0=xinit)
    x2 = res2['x'][0]
    res3 = minimize(o3, x0=xinit)
    x3 = res3['x'][0]
    
    x1l[i] = x1
    x2l[i] = x2
    x3l[i] = x3
    g = -x1 + 2*x2 - x3 - 2
    signG[i] = bool(g>0)
    lam += sum([rho1,rho2,rho3])/6 * g #(rho3*rho1+(1-rho3)*rho2) * g
    l[i] = lam
    
    if lam < 0:
        lam  = 0
#    f1 = lambda x: -x1 + 2*x-g-2
#    f1 = lambda x: 2*x2 - x -2
#    
#    plt.plot(t,f1(t), 'b-')
#    plt.plot(x2,x3, 'rp', markersize = 1)

#fl = lambda x: -x1 + 2*x - 2
#plt.plot(t,fl(t),'r-')

#plt.plot(x2,x3, 'rp', markersize = 5)
print(x1,x2,x3,g)
#plt.figure()
#plt.plot(l)
plt.figure()
plt.plot(x1l, label='x1')
plt.plot(x2l,label='x2')
plt.plot(x3l,label='x3')




#def running_mean(x, N):
#    cumsum = np.cumsum(np.insert(x, 0, 0)) 
#    return (cumsum[N:] - cumsum[:-N]) / float(N)
#
##plt.figure()
#a = np.cumsum(x1l)/np.arange(1,n+1)
#b = np.cumsum(x2l)/np.arange(1,n+1)
##plt.plot(a)
##plt.plot(b)
#
#plt.figure()
#plt.plot(np.abs(a - 2.4))
#plt.plot(np.abs(b- 2.2))
#
#
#print(a[-1], b[-1])
#plt.figure()
#plt.plot(signG)
#plt.legend()

#o1 = lambda x: (x - 1)**2 
#o2 = lambda x: (x - 1)**2 - lam*x 
##    o2 = lambda x: (x-5)**2 + 2*lam*x
##    o3 = lambda x: (x-2)**2 - lam*x
#res1 = minimize(o1, x0=xinit)#, bounds=bnds)#, constraints=cons)
#res2 = minimize(o2, x0=xinit)
#
#t = np.linspace(-5,5,100)
#x = o1(t)
#x2 = o2(t)
#
#plt.plot(t,x)
#plt.plot(t,x2)
#
#print(res1['x'])
#print(res2['x'])