# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:47:14 2018

@author: kst
"""
import numpy as np
from threading import Thread
import FFArithmetic as field
import shamir_scheme as ss
from scipy.optimize import minimize
import proc
import time

class server:
    securecom = {}   #cloud internal communication
    broadcasts = {}

    def __init__(self,F, n, t, numTrip, l = 7):
        self.b = ss.share(F,np.random.choice([-1,1]), t, n)
        self.triplets = [proc.triplet(F,n,t) for i in range(numTrip)]
        self.r, self.rb = proc.randomBitsDealer(F,n,t,l)
        
class cloud(Thread):
    
    def __init__(self, F, x, n, t, i, s):
        Thread.__init__(self)
        self.c = 0
        self.comr = 0
        self.F = F
        self.x = x
        self.n = n
        self.t = t
        self.i = i
        self.server = s
        self.comtime = 0
        
    def distribute_shares(self):
        shares = ss.share(self.F, self.x, self.t, self.n)
        s = 'x' + str(self.i)
        st = time.time()
        self.server.securecom[s] = shares
        sl = time.time()
        self.comtime +=(sl-st)
        
    def get_share(self, a):
        st = time.time()
        while True:
            try:
                res =  (self.server.securecom[a][self.i])
                break
            except:
                continue
        sl = time.time()
        self.comtime +=(sl-st)
        return res
    
    def get_broadcast(self, a):
        st = time.time()
        while True:
            try:
                res = self.server.broadcasts[a]
                break
            except:
                continue
        sl = time.time()
        self.comtime += (sl-st)
        return res
    def broadcast(self, name, a):
        st = time.time()
        self.server.broadcasts[name + str(self.i)] = a
        sl = time.time()
        self.comtime += sl-st
    def reconstruct_secret(self, a):
        res = []
        for i in range(self.n):
            res.append(self.get_broadcast(a + str(i)))
        return ss.rec(self.F, res)
    
    def add_shares(self,a,b):
        return a + b
    
    def mult_shares(self, a, b):
        r = self.server.triplets[self.c][self.i]
        self.c += 1
        
        d_local = a - r[0]
        self.broadcast('d' + str(self.comr), d_local)
        d_pub = self.reconstruct_secret('d' + str(self.comr))
        self.comr +=1
        e_local = b - r[1]
        self.broadcast('e' + str(self.comr), e_local)
        e_pub = self.reconstruct_secret('e' + str(self.comr))
        self.comr+=1
        return d_pub * e_pub + d_pub*r[1] + e_pub*r[0] + r[2]
        
    
    def run(self):
## DISTRIBUTE INPUT
        self.distribute_shares()
## GET INPUT SHARINGS FROM ALL PARTIES
        input_shares = []
        for i in range(self.n):
            input_shares.append(self.get_share('x' + str(i)))
## BIT COMPARISON
#        res = self.bitDecLess(input_shares[0], input_shares[1], 7)
#        self.broadcast('r', res)
#        print(self.reconstruct_secret('r'))
#        c = self.legendreComp(input_shares[0], input_shares[1])

## Find minimum:
#        for i in range(0,self.n-1):
#            if i == 0:          
#                c = self.legendreComp(input_shares[0], input_shares[1])
#                self.broadcast('cb', c)
#                                     
#                a = self.mult_shares(1-c,input_shares[0]) + self.mult_shares(c,input_shares[1])
#                self.broadcast('aa', a)         
#
#            else:
#                c = self.legendreComp(a, input_shares[i+1])
#                a = self.mult_shares(1-c,a)+self.mult_shares(c,input_shares[i+1])
#            
#        
#        self.broadcast('c2', c)
#        print(self.reconstruct_secret('c2'))
        
## MULTIPLICATION
#        a = self.mult_shares(input_shares[0], input_shares[1])
#        b = self.mult_shares(a, input_shares[2])
#        c = self.mult_shares(b, input_shares[3])
#        self.broadcast('p', c)
#        print(self.reconstruct_secret('p'))
        
#SUMMATION OF SECRETS
        sum_share = sum(input_shares)
        self.broadcast('s', sum_share)
        print(self.reconstruct_secret('s'))       
#     
        
class Agent(Thread):
    
    def __init__(self, F, n, t, obj, lam, ite, name):
        Thread.__init__(self)
        self.c = 0
        self.comr = 0
        self.F = F
        self.n = n
        self.t = t
        self.obj = obj
        self.lam = lam
        self.x = None
        self.iterations = ite
        self.name = name
        
    def distribute_shares(self, name):
        shares = ss.share(self.F, self.x, self.t, self.n)
        s = name + str(self.name)
        self.server.securecom[s] = shares
    
    def get_share(self, a):
        while True:
            try:
                res =  (self.server.securecom[a][self.i])
                break
            except:
                continue
        return res
    def get_broadcast(self, a):
        while True:
            try:
                res = self.server.broadcasts[a]
                del self.server.broadcasts[a]
                break
            except:
                continue
        return res
    
    def reconstruct_secret(self, a, m = 1):
        res = m*[[]]
        for i in range(self.n):
            temp = self.get_broadcast(a + str(i))
            for k in range(m):
                res[k].append(temp[k])
        res1 = []
        for i in range(m):
            res1.append(ss.rec(self.F, res[i]))
        return res1
    
    def run(self):
        xinit = [0]
        for i in range(self.iterations):
            c1, c2 = self.reconstruct_secret(str(self.name), 2)
            
            obj1 = lambda x: c1 * x**2 + c2 * x
            
            obj = lambda x, f = obj1, l = self.obj: l(x) + f(x)
            
            res = minimize(obj, x0=xinit)#, bounds=bnds)#, constraints=cons)
            self.x = res['x'][0]
            
            
        
        
        
        
        
        
        
        
        
        
F = field.GF(7979490791)            
n = 3
t = 1
serv = server(F,n,t, 300)
p1 = party(F,2,n,t,0,serv)
p2 = party(F,3,n,t,1,serv)
p3 = party(F,5,n,t,2,serv)
threads = [p1,p2,p3]
#p4 = party(F,1,n,t,3,serv)
start = time.time()
p1.start()
p2.start()
p3.start()
#p4.start()

for t in threads:
    t.join()

end = time.time()
ex = end-start
print('Execution time: ', ex)

print(p1.comtime)
print(p2.comtime)
print(p3.comtime)
print('\n')
print(p1.comtime/ex)
print(p2.comtime/ex)
print(p3.comtime/ex)
