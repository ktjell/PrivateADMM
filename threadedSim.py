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
import matplotlib.pyplot as plt

class server:
    securecom = {}   #cloud internal communication
    broadcasts = {}
    A1 = []
    A2 = []
    A3 = []
    A11 = []
    A22 = []
    A33 = []
    A1l = []
    A2l = []
    A3l = []

    def __init__(self,F, n, t, numTrip, l = 7):
#        self.b = ss.share(F,np.random.choice([-1,1]), t, n)
        self.triplets = [proc.triplet(F,n,t) for i in range(numTrip)]
#        self.r, self.rb = proc.randomBitsDealer(F,n,t,l)
        
class cloud(Thread):
    
    def __init__(self, F, n, a, t, i, s, H, b, ite, rho):
        Thread.__init__(self)
        self.c = 0
        self.comr = 0
        self.F = F #Finite Field (from FFArithmetic class)
        self.n = n #number of cloud servers
        self.a = a #number of agents
        self.t = t #number of corrupted cloud servers
        self.i = i #cloud-server number
        self.server = s #used to simulate communication (not "secure" but will be in a real imlementation)
        self.comtime = 0
        self.H = H #constraints
        self.b = b #constraints
        self.iterations = ite
        self.rho = rho
#    def distribute_shares(self):
#        shares = ss.share(self.F, self.x, self.t, self.n)
#        s = 'x' + str(self.i)
#        st = time.time()
#        self.server.securecom[s] = shares
#        sl = time.time()
#        self.comtime +=(sl-st)
        
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
    
    def innerProd(self, a, b):
        s = 0
        for i in range(len(a)):
            s += self.mult_shares(a[i], b[i])
        return s
    
    def run(self):
## GET INPUT SHARINGS FROM ALL AGENTS
        #TODO: Implementer random matrix T for at gøre constrains mere hemmelige
        #TODO: Implementer at x_i er en vector og ikke scalar
        lam = len(self.H)*[0]
        for j in range(self.iterations):
            
        
            input_shares = []
            for i in range(self.a):
                input_shares.append(self.get_share('x' + str(j) + str(i)))

            #TODO: Lav rho hemmelig 
            #Calculate lambda^i
            for i in range(len(self.H)):
                cons = self.innerProd(self.H[i], input_shares) + 100*self.b[i]
                lam[i] += self.rho * cons 
                self.broadcast('lam'+str(i) + str(self.comr), lam[i])
                
                if self.i == 0:
                    self.server.A1l.append(self.reconstruct_secret('lam'+str(i)+ str(self.comr)))
                if self.i == 1:
                    self.server.A2l.append(self.reconstruct_secret('lam'+str(i)+ str(self.comr)))
                if self.i == 2:
                    self.server.A3l.append(self.reconstruct_secret('lam'+str(i)+ str(self.comr)))
                self.comr+= 1
            #Calculate c1
            c1 = []
            for i in range(self.a):
                s = 0
                for k in range(len(self.H)):
                    s += self.mult_shares(self.H[k][i],self.H[k][i])
                c1.append(self.rho * s)
                
                
            c2 = []
            for i in range(self.a):
                o = 0
                for k in range(len(self.H)):
                    o += lam[k] * self.H[k][i]
                    for g in range(len(self.H[0])):
                        if g != i:
                            o +=  self.rho * self.mult_shares(self.mult_shares(self.H[k][g], self.H[k][i]), input_shares[g])
                c2.append(o)
                         
            for i in range(self.a):
                self.server.broadcasts['c'+ str(i) + str(self.i)] = [c1[i], c2[i]]
            
#            self.comr = 0
            self.c = 0
            
class agent(Thread):
    
    def __init__(self, F, n, t, obj, ite, name, serv):
        Thread.__init__(self)
        self.c = 0
        self.comr = 0
        self.F = F
        self.n = n
        self.t = t
        self.obj = obj
        self.x = None
        self.iterations = ite
        self.name = name
        self.server = serv
        
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
        res = [[], []]
        for i in range(self.n):
            temp = self.get_broadcast(a + str(i))
            for k in range(m):
                res[k].append(temp[k])
        res1 = []
        for i in range(m):
            res1.append(ss.rec(self.F, res[i]))
        return res1, res
    
    def run(self):
        
        xinit = [0]
        for i in range(self.iterations):
            if i == 0:
                c1, c2 = [0,0]
            else:
                [c1, c2], res = self.reconstruct_secret('c' + str(self.name), m = 2)
            c1 = float(str(c1))
            c2 = float(str(c2))
            if c1 > 1000000:
                c1 = -float(str(ss.rec(self.F, -1*np.array(res[0]))))
            if c2 > 1000000:
                c2 = -float(str(ss.rec(self.F, -1*np.array(res[1]))))

#            print('{}: {}'.format(self.name, [float(str(c1))/200.,float(str(c2))/100.]))
            obj1 = lambda x: c1/200 * x**2 + c2/10000 * x
            
            obj = lambda x, f = obj1, l = self.obj: l(x) + f(x)
            
            res = minimize(obj, x0=xinit)#, bounds=bnds)#, constraints=cons)
            self.x = res['x'][0]
            
            
            #print('Result from agent {}: {}'.format(self.name, self.x))
#            if self.name == '0':
#                self.server.A1.append([float(str(c1))/200.,float(str(c2))/10000.])
#            if self.name == '1':
#                self.server.A2.append([float(str(c1))/200.,float(str(c2))/10000.])
#            if self.name == '2':
#                self.server.A3.append([float(str(c1))/200.,float(str(c2))/10000.])
            
            if self.name == '0':
                self.server.A11.append(self.x)
            if self.name == '1':
                self.server.A22.append(self.x)
            if self.name == '2':
                self.server.A33.append(self.x)
            self.x=int(100*self.x)
            self.distribute_shares('x'+str(i))
        
        
        
##################### MAIN ################################
pp = 7979490791
F = field.GF(pp)            
n = 3
t = 1
a = 3
H = [[ss.share(F,-1,t,n), ss.share(F,2,t,n), ss.share(F,-1,t,n)], [ss.share(F,5,t,n), ss.share(F,7,t,n), ss.share(F,-3,t,n)]]
b = [ss.share(F,-2,t,n), ss.share(F,-5,t,n)]

Hs= []
for j in range(n):
    temp = []
    for i in range(2):
        temp.append([H[i][0][j], H[i][1][j], H[i][2][j]])
    Hs.append(temp)

bs = []
for i in range(n):
    bs.append([b[0][i],b[1][i]])

ite = 50
rho = 1
serv = server(F,n,t, 300)
obj1 = lambda x: (x - 1)**2
obj2 = lambda x: (x - 5)**2
obj3 = lambda x: (x - 2)**2


p1 = cloud(F, n, a, t, 0, serv, Hs[0], bs[0], ite-1, rho)
p2 = cloud(F, n, a, t, 1, serv, Hs[1], bs[1], ite-1, rho)
p3 = cloud(F, n, a, t, 2, serv, Hs[2], bs[2], ite-1, rho)
a1 = agent(F, n, t, obj1, ite, '0', serv)
a2 = agent(F, n, t, obj2, ite, '1', serv)
a3 = agent(F, n, t, obj3, ite, '2', serv)

threads = [p1,p2,p3]
agents = [a1,a2,a3]

for a in agents:
    a.start()

for k in threads:
    k.start()

#
for a in agents:
    a.join()

#ka = zip(serv.A1, serv.A2, serv.A3)
#for i in ka:
#    print('c1,c2', i)
#
#
#
#
print('A1: \n')
for i in range(ite):
    print('({},{})'.format(i, serv.A11[i]))

print('A2: \n')
for i in range(ite):
    print('({},{})'.format(i, serv.A22[i]))

print('A3: \n')
for i in range(ite):
    print('({},{})'.format(i, serv.A33[i]))

#    

plt.plot(serv.A11)
plt.plot(serv.A22)
plt.plot(serv.A33)

#
#
#ka = zip(serv.A1l, serv.A2l, serv.A3l)
#for i in ka:
#    print('lam', i)
