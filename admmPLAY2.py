# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:22:49 2019

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
x1 = 0
x2 = 0
x3 = 0
rho1 = .01#np.random.uniform(0.2,1)
rho2 = .01#np.random.uniform(0.2,1)
rho3 = .01#np.random.uniform(0.2,1)
for i in range(n):
    p1 = rho1/2
    p2 = rho2/2
    p3=  rho3/2
    c1 =  np.array([p1*26, -lam1 + 5*lam2 + p1*(66*x2 -28*x3-46)])
    c2 =  np.array([p2*53, 2*lam1 + 7*lam2 +p2*(66*x1 -46*x3-74)])
    c3 =  np.array([p3*10, p3*(-28*x1 -46*x2+34) -lam1 -3*lam2])
    if i == 0:
        c1,c2,c3 = [[0,0],[0,0],[0,0]]
#    print('c1,c2', c1, c2, c3)
    
#    plt.plot(0,c1[0], 'bo')
#    plt.plot(1,c1[1], 'ro')
#    print('(5,{})'.format(c3[1]))
#    plt.plot(2,c1[2], 'go')
#    plt.plot(3,c1[3], 'yo')
#    plt.plot(4,c1[4], 'mo')
    o1 = lambda x: (x-1)**2  + c1[0] * x**2 + c1[1]*x # <--------------
#    p=p1
#    o1 = lambda x: (x-1)**2  -lam1*x+5*lam2*x   +p*x**2-p*4*x2*x+p*2*x3*x+p*4*x+p*25*x**2+p*70*x2*x-p*30*x3*x-50*p*x
#    o1 = lambda x: (x-1)**2  +(-lam1+5*lam2)*x   +(26)*p*x**2  + p*x*(66*x2  - 28*x3 - 46 ) 
    
#    o2 = lambda x: (x-5)**2  + c2[0] * x**2 + c2[1]*x
#    o2 = lambda x: (x-5)**2  +2*lam1*x+7*lam2*x +p*4*x**2-p*4*x1*x-4*p*x3*x-p*4*x+p*49*x**2+p*70*x1*x-p*42*x3*x-p*70*x
    o2 = lambda x: (x-5)**2    + c2[0]*x**2  + x*c2[1]#(p2*(66*x1  -46*x3  -74) + (2*lam1+7*lam2)) #<-----------------
#    o21 = lambda x: (x-5)**2    + c2[0]*x**2  + x*(p2*(66*x1  -46*x3  -74) + (2*lam1+7*lam2))
#    o3 = lambda x: (x-2)**2    + c3[0]*x**2    +x* c3[1]
    
#    print (p3*(-28*x1 -46*x2 + 34) + (-lam1-3*lam2) == c3[1])
#    a = -p3*28*x1 -p3*46*x2 + p3*34 -lam1-3*lam2
    
#    o3 = lambda x: (x-2)**2    + c3[0]*x**2    + x*(-p3*28*x1 -p3*46*x2 + p3*34 -lam1-3*lam2) #((p3*(-28*x1 -46*x2 + 34) + (-lam1-3*lam2)))
#    o31 = lambda x:(x-2)**2    + c3[0]*x**2    + x*a

    o3 = lambda x: (x-2)**2    + c3[0]*x**2    + x*c3[1]#*(-p3*28*x1 -p3*46*x2 + p3*34 -lam1-3*lam2) #((p3*(-28*x1 -46*x2 + 34) + (-lam1-3*lam2)))
#    o31= lambda x: (x-2)**2    + c3[0]*x**2    + x*a
    
    res1 = minimize(o1, x0=xinit)#, bounds=bnds)#, constraints=cons)
    x1 = res1['x'][0]
    
    res2 = minimize(o2, x0=xinit)
    x2 = res2['x'][0]


    res3 = minimize(o3, x0=xinit)
    x3 = res3['x'][0]
#    print('res', x1,x2,x3)
    x1l[i] = x1
    x2l[i] = x2
    x3l[i] = x3
    g1 = -x1 + 2*x2 - x3 - 2
    lam1 += (rho1+rho2+rho3)/3 * g1
    l[i] = lam1
    
    g2 = 5*x1 + 7*x2 - 3*x3 - 5
    lam2 += (rho1+rho2+rho3)/3 * g2
    l2[i] = lam2
    
    if lam1 < 0:
        lam1  = 0
#    print('lam', lam1*100,lam2*100)
#    f1 = lambda x: -x1 + 2*x-g-2
#    f1 = lambda x: 2*x2 - x -2
    
#    plt.plot(t,f1(t), 'b-')
#    plt.plot(x2,x3, 'rp', markersize = 1)

#fl = lambda x: -x1 + 2*x - 2
#plt.plot(t,fl(t),'r-')

#plt.plot(x2,x3, 'rp', markersize = 5)
#print(x1,x2,x3,g1)
#plt.figure()
#plt.plot(l)
#plt.figure()
#plt.plot(x1l, label='x1')
#plt.plot(x2l,label='x2')
#plt.plot(x3l,label='x3')
#plt.show()


print('x1: \n')
for i in range(n):
    print('({},{})'.format(i, x1l[i]))

print('A2: \n')
for i in range(n):
    print('({},{})'.format(i, x2l[i]))

print('A3: \n')
for i in range(n):
    print('({},{})'.format(i, x3l[i]))

#print(x1,x2,x3)

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