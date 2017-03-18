
import sys
import os
import scipy
import cPickle
import math
import numpy as np
from numpy.random import seed,random_integers,rand,permutation
from functools import partial
Dim = 7

def set_dim(d):

 setattr(sys.modules[__name__],"Dim",d)

class sim_ann:

 def __init__(self,f,T0,alpha,P,L):
  seed()
  self.f = f
  self.s = scipy.array([5.-10.*rand() for i in range(Dim)])
  self.T = T0
  self.fit = self.f(self.s)
  self.hall_of_fame = []   
  self.P = int(P)
  self.L = int(L)
  self.alpha = alpha
  for i in scipy.arange(5):
   self.hall_of_fame.insert(0,scipy.hstack((self.fit,self.s)))
  
 def Perturba(self,x):
  for i in range(x.shape[0]):
   if scipy.rand() < 0.3:
    aux = x[i]
    x[i] = x[i] + 0.6*scipy.randn()
    if not (-5. <= x[i] <= 5.):
	 x[i] = aux
  return x
  
 def run(self):
  i = 1
  self.nS = 0
  while (True):
   si = self.Perturba(self.s.copy())
   aux = self.f(si)
   delta = aux - self.fit
   if (delta < 0) or (math.exp(-delta/self.T) > scipy.rand()):
    self.s = si.copy()
    self.fit = aux
    self.nS = self.nS + 1
   i = i + 1
   if (i > self.P) or (self.nS > self.L):
	k = 0
	if self.nS > 0:
	 while (self.fit > self.hall_of_fame[k][0]):
	  k = k + 1
	  if k == 5:
	   break
	 if k < 5:
	  self.hall_of_fame.insert(k,scipy.hstack((self.fit,self.s)))
	  self.hall_of_fame.pop()   
	break
  self.T = self.alpha*self.T
  
class coevol:
 def __init__(self,fitness_func,ns1 = 10,ns2 = 10,npop1 = 20,pr = 0.3,beta = 0.7,npop2 = 20,w = 0.75,c1 = 1.5,c2 = 1.5,delta = 0.4,alpha = 1.):
  seed()
  self.ns1 = ns1
  self.ns2 = ns2
  self.ff = fitness_func
  f1 = partial(self.f,pop = 5-10.*rand(ns2,Dim),hf = 5-10.*rand(5,Dim),ff = self.ff)
  f2 = partial(self.f,pop = 5-10.*rand(ns1,Dim),hf = 5-10.*rand(5,Dim),ff = self.ff)
  self.p1 = de(f1,npop1,pr,beta)
  self.p2 = pso(f2,npop2,w,c1,c2,delta,alpha)
  #self.p2 = de(f2,npop1,pr,beta)
  
  self.hall_of_fame1 = []
  de_best = self.p1.pop[self.p1.fit.argmin()] 
  for i in scipy.arange(5):
    self.hall_of_fame1.insert(0,scipy.hstack((self.ff(de_best),de_best)))
	
  #self.hall_of_fame2 = []	
  #de_best = self.p2.pop[self.p2.fit.argmin()] 	
  #for i in scipy.arange(5):
  #  self.hall_of_fame2.insert(0,scipy.hstack((self.ff(de_best),de_best)))
	
  self.hall_of_fame2 = []
  pso_best = self.p2.pop[self.p2.fit.argmin()]
  for i in scipy.arange(5):
	self.hall_of_fame2.insert(0,scipy.hstack((self.ff(pso_best),pso_best)))
  
 def f(self,x,pop,hf,ff):
  #if (x > 5.).any() or (x < -5.).any():
  # score_x = 500
  #else:
  # score_x = 0
  score_x = 0
  ans_x = ff(x)
  ans_pop = scipy.array([ff(y) for y in pop])
  score_x  = score_x + 4*math.tanh(0.5*(ans_x - ans_pop.mean())) 
  score_x  = score_x + 10*math.tanh(0.5*(ans_x - hf[:,0].mean())) 
  return score_x 
  #for y in ans_pop:
   #score_x = score_x + 0.15*(ans_x - y)*math.exp(0.95*(ans_x - y))
   #score_x = score_x + (2*(ans_x - y))**2*scipy.tanh(2*(ans_x - y))
  #for y in hf[:,0]:
   #score_x = score_x + 0.85*(ans_x - y)*math.exp(0.95*(ans_x - y))
   #score_x = score_x + 2*(2*(ans_x - y))**2*scipy.tanh(2*(ans_x - y))
  #return score_x
  
 def HF_Updt(self,hf,x,y):
  # Hall of fame
  k = 0
  while (x > hf[k][0]):
   k = k + 1
   if k == 5:
    break
  if k < 5 and not (x == hf[k][0]):
   hf.insert(k,scipy.hstack((x,y)))
   hf.pop()
   
 def run(self):
  i = permutation(len(self.p2.pop))[0:self.ns2]
  f1 = partial(self.f,pop = self.p2.pop[i],hf = scipy.array(self.hall_of_fame2),ns = self.ns2,ff = self.ff)
  self.p1.fitness_func = f1   
  self.p1.run()
  best = self.p1.pop[self.p1.fit.argmin()]
  self.HF_Updt(self.hall_of_fame1,self.ff(best),best)
  i = permutation(len(self.p1.pop))[0:self.ns1]
  f2 = partial(self.f,pop = self.p1.pop[i],hf = scipy.array(self.hall_of_fame1),ff = self.ff)
  self.p2.fitness_func = f2 
  self.p2.run()
  #best = self.p2.pop[self.p2.fit.argmin()]
  #self.HF_Updt(self.hall_of_fame2,self.ff(best),best)
  self.HF_Updt(self.hall_of_fame2,self.ff(self.p2.bfg),self.p2.bfg)  	   

class de:

 def __init__(self,fitness_func,npop = 10,pr = 0.7,beta = 2.5):
  seed()
  self.ns = int(npop)
  self.beta = beta
  self.pr  = pr 
  self.ff = fitness_func
  self.pop = scipy.array([self.gera_individuo() for i in range(self.ns)])
  self.fit = scipy.array([self.ff(i) for i in self.pop])

 def gera_individuo(self):

   return 5 - 10.*rand(Dim) 
    
 def run(self):  
   
  for i in scipy.arange(self.ns):
   # para cada individuo da populacao 
   # gera trial vector usado para perturbar individuo atual (indice i)
   # a partir de 3 individuos escolhidos aleatoriamente na populacao e
   # cujos indices sejam distintos e diferentes de i
   invalido = True
   while invalido:
    j = random_integers(0,self.ns-1,3)
    invalido = (i in j)
    invalido = invalido or (j[0] == j[1]) 
    invalido = invalido or (j[1] == j[2]) 
    invalido = invalido or (j[2] == j[0])
   
   # trial vector a partir da mutacao de um alvo 
   u = self.pop[j[0]] + self.beta*(self.pop[j[1]] - self.pop[j[2]])
 
   # gera por crossover solucao candidata
   c = self.pop[i].copy()  
   # seleciona indices para crossover
   # garantindo que ocorra crossover em
   # pelo menos uma vez
   j = random_integers(0,self.pop.shape[1]-1)
  
   for k in scipy.arange(self.pop.shape[1]):
    if (scipy.rand() < self.pr) or (k == j):
     c[k] = u[k]  

   c_fit = self.ff(c) 
   self.fit[i] = self.ff(self.pop[i])
       
   # leva para proxima geracao quem tiver melhor fitness
   if c_fit < self.fit[i]:
    self.pop[i] = c
    self.fit[i] = c_fit
 
class pso:

 def __init__(self,fitness_func,npop = 20,w = 0.5,c1 = 2.01,c2 = 2.02,delta = 0.65,alpha = .8):
  seed()
  self.c1 = c1
  self.c2 = c2
  self.w = w
  self.ns = int(npop) 
  self.ff = fitness_func
  self.vmax = delta*10.
  self.it = 0
  self.alpha = alpha
  self.pop = scipy.array([self.gera_individuo() for i in range(self.ns)])
   # avalia fitness de toda populacao
  self.fit = scipy.array([self.ff(i) for i in self.pop])  
   # inicializa velocidades iniciais
  self.v = scipy.zeros((self.ns,Dim))
   # guarda a melhor posicao de cada particula 
  self.bfp = scipy.copy(self.pop)
  self.bfp_fitness = scipy.copy(self.fit)
   # guarda a melhor posicao global
  self.bfg = self.pop[self.bfp_fitness.argmin()].copy()
  self.bfg_fitness = self.bfp_fitness.min().copy()
       
 def gera_individuo(self):
  return 5-10.*rand(Dim)
  
 def run(self):
  self.bfg_fitness = self.ff(self.bfg)
  for i in scipy.arange(self.ns):
   self.bfp_fitness[i] = self.ff(self.bfp[i])
   # Atualiza velocidade
   self.v[i] = self.w*self.v[i] 
   self.v[i] = self.v[i] + self.c1*scipy.rand()*( self.bfp[i] - self.pop[i]) 
   self.v[i] = self.v[i] + self.c2*scipy.rand()*(self.bfg - self.pop[i])
   self.v[i] = self.vmax*scipy.tanh(self.v[i]/self.vmax)
   self.pop[i] = self.pop[i] + self.v[i]    
   self.fit[i] = self.ff(self.pop[i])
   # Atualiza melhor posicao da particula
   if self.fit[i] < self.bfp_fitness[i]:
	self.bfp[i] = self.pop[i].copy()
	self.bfp_fitness[i] = self.fit[i]
	# Atualiza melhor posicao global
	if  self.bfp_fitness[i] < self.bfg_fitness:
		self.bfg_fitness = self.bfp_fitness[i].copy()
		self.bfg = self.bfp[i].copy()
  self.vmax = self.vmax*self.alpha   
  		
		
#############################
## Some Benchmark functions #
#############################
#Minimo = 0 em (1,1)
def f1(x):
 aux = 0
 for i in scipy.arange(x.shape[0]-1):
  aux = aux + 100*(x[i+1] - x[i]**2)**2+(x[i]-1)**2 
 return aux 

# Minimo = -1 em (0,0) 
def f2(x):
 xx = x[0]**2+x[1]**2 
 aux = 1 + scipy.cos(12*scipy.sqrt(xx))
 aux = aux / (.5*xx+2) 
 return -aux

# Minimo = 0 em (0,0) 
def f3(x):
  k1 = 0
  k2 = 0
  k3 = 1/float(x.shape[0])
  for i in range(x.shape[0]):
   k1 = k1 + x[i]**2
   k2 = k2 + scipy.cos(2*scipy.pi*x[i])
   
  return -20*scipy.exp(-0.2*scipy.sqrt(k3*k1))-scipy.exp(k3*k2)+20+scipy.exp(1)
 
