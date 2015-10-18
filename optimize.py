import sys
import scipy
import math
import cPickle
import numpy as np
from numpy.random import random_integers,rand,permutation

Dim = 7

def set_dim(d):
 Dim = d

class sim_ann:

 def __init__(self,f,s0,T0,alpha,P,L):
  self.s = s0
  self.T = T0
  self.P = P
  self.L = L
  self.alpha = alpha
  self.f = f
  self.fit = self.f(self.s)
  self.hall_of_fame = []   
  for i in scipy.arange(5):
   self.hall_of_fame.insert(0,scipy.hstack((self.fit,self.s)))
  
 def Perturba(self,x,f):
  for i in range(x.shape[0]):
   if scipy.rand() < 0.3:
    aux = x[i]
    x[i] = x[i] + 0.6*x[i]*(1+f)*scipy.randn()
    if not (0.125 <= x[i] <= 125):
	 x[i] = aux
  return x
  
 def run(self):
  i = 1
  self.nS = 0
  while (True):
   si = self.Perturba(self.s.copy(),self.fit)
   aux = self.f(si)
   delta = aux - self.fit
   if (delta < 0) or (math.exp(-delta/self.T) > scipy.rand()):
    self.s = si.copy();self.fit = aux
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

 def __init__(self,challenge_func,ns = 10,npop1 = 40,pr = 0.3,beta = 0.5,npop2 = 100,w = 0.5,c1 = 2.01,c2 = 2.02):
  # Tamanho das populacoes
  self.ns = ns
  self.npop1 = npop1
  self.npop2 = npop2
  # Parametros do DE
  self.beta = beta
  self.pr  = pr
  # Parametros do PSO
  self.c1 = c1
  self.c2 = c2
  self.w = w 
  # Funcao que representa problema desafio  
  self.fc = challenge_func
  # Respostas do problema desafio
  #self.pso = pso(fitness_func = challenge_func,npop = npop2,w = w,c1 = c1,c2 = c2)
  self.ans1 = scipy.zeros(self.npop1)
  self.ans2 = scipy.zeros(self.npop2)
  # Populacoes
  self.pop1 = []
  self.pop2 = []
  # Gera pop1 e pop2 e resolve problema desafio
  for i in scipy.arange(self.npop1):
   self.ans1[i],aux = self.resolve_desafio(self.gera_individuo())
   self.pop1.append(aux.copy())
   
  for i in scipy.arange(self.npop2):
   self.ans2[i],aux = self.resolve_desafio(self.gera_individuo())
   self.pop2.append(aux.copy())
    
  self.pop1 = scipy.array(self.pop1)
  self.pop2 = scipy.array(self.pop2)
  
  self.hall_of_fame1 = []
  for i in scipy.arange(5):
    self.hall_of_fame1.insert(0,scipy.hstack((self.ans1.min(),self.pop1[self.ans1.argmin()])))
	
  self.hall_of_fame2 = []
  for i in scipy.arange(5):
    #self.hall_of_fame2.insert(0,scipy.hstack((self.pso.fit[0],self.pso.pop[0])))
	self.hall_of_fame2.insert(0,scipy.hstack((self.ans2.min(),self.pop2[self.ans2.argmin()])))
	
  # Funcoes fitness das populacoes
  self.fit1 = scipy.zeros(self.npop1)
  self.fit2 = scipy.zeros(self.npop2)
  
  for i in scipy.arange(self.npop2):
    self.fit2[i] = self.avalia_aptidao2(self.ans2[i])

  for i in scipy.arange(self.npop1):
    self.fit1[i] = self.avalia_aptidao1(self.ans1[i])		
    
  # inicializa velocidades iniciais do PSO
  self.v = scipy.array([self.gera_individuo() for i in scipy.arange(self.npop2)])
  # guarda o melhor fitness de cada particula PSO 
  self.bfp = scipy.copy(self.pop2)
  self.bfp_fitness = scipy.copy(self.fit2)
  self.bfp_ans = scipy.copy(self.ans2) 
  # guarda o melhor fitness global PSO
  self.bfg = self.pop2[self.bfp_fitness.argmax()].copy()
  self.bfg_fitness = self.bfp_fitness.max().copy()
  self.bfg_ans = self.bfp_ans[self.bfp_fitness.argmax()].copy()
  

 def gera_individuo(self):
   #return np.array([124.8*rand()+0.2 for i in range(Dim)])
   return 125.*rand(Dim)+0.125
   
 def resolve_desafio(self,x):
  for i in range(x.shape[0]):
   if not 0.125 <= x[i] <= 125.125:
    x[i] = 125.*rand()+0.125
    #x[i] = 50. - 100.*rand()
	
  return (self.fc(x),x)
  
 def avalia_aptidao2(self,x):
  cnt = 0.
  k = self.ns
  i = permutation(self.npop1)[0:k]
  for a in self.ans1[i]:  
   if x < a:
        cnt = cnt + 20*(a - x)	
  for a in scipy.array(self.hall_of_fame1)[:,0]:
   if x < a:
        cnt = cnt + 40*(a - x)
  return cnt
 
 def avalia_aptidao1(self,x):
  cnt = 0
  k = self.ns
  i = permutation(self.npop2)[0:k]
  for a in self.ans2[i]:  
   if x<a:
        cnt = cnt + 20*(a - x)
  for a in scipy.array(self.hall_of_fame2)[:,0]:
   if x<a:
        cnt = cnt + 40*(a - x)
  return cnt

 def HF1_Updt(self,x,y):
  # Hall of fame
  k = 0
  while (x > self.hall_of_fame1[k][0]):
   k = k + 1
   if k == 5:
    break
  if k < 5 and not (x == self.hall_of_fame1[k][0]):
   self.hall_of_fame1.insert(k,scipy.hstack((x,y)))
   self.hall_of_fame1.pop()

 def HF2_Updt(self,x,y):
  # Hall of fame
  k = 0
  while (x > self.hall_of_fame2[k][0]):
   k = k + 1
   if k == 5:
    break
  if k < 5 and not (x == self.hall_of_fame2[k][0]):
   self.hall_of_fame2.insert(k,scipy.hstack((x,y)))
   self.hall_of_fame2.pop()
   
 def Evolve_DE(self):
  for i in scipy.arange(self.npop1):
   # para cada individuo da populacao 
   # gera trial vector usado para perturbar individuo atual (indice i)
   # a partir de 3 individuos escolhidos aleatoriamente na populacao e
   # cujos indices sejam distintos e diferentes de i
   invalido = True
   while invalido:
    j = random_integers(0,self.npop1-1,3)
    invalido = (i in j)
    invalido = invalido or (j[0] == j[1]) 
    invalido = invalido or (j[1] == j[2]) 
    invalido = invalido or (j[2] == j[0])    
   # trial vector a partir da mutacao de um alvo 
   u = self.pop1[j[0]] + self.beta*(self.pop1[j[1]] - self.pop1[j[2]]) 
   # gera por crossover solucao candidata
   c = self.pop1[i].copy()  
   # seleciona indices para crossover
   # garantindo que ocorra crossover em
   # pelo menos uma vez                 
   j = random_integers(0,self.pop1.shape[1]-1)
   for k in scipy.arange(self.pop1.shape[1]):
    if (scipy.rand() < self.pr) or (k == j):
     c[k] = u[k]  
   ans,c = self.resolve_desafio(c)
   c_fit = self.avalia_aptidao1(ans)    
   # leva para proxima geracao quem tiver melhor fitness
   if (c_fit > self.fit1[i]):
    self.pop1[i] = c
    self.fit1[i] = c_fit
    self.ans1[i] = ans
	
 def Evolve_PSO(self):
  for i in scipy.arange(self.npop2):
   # Atualiza velocidade
   self.v[i] = self.w*self.v[i] 
   self.v[i] = self.v[i] + self.c1*scipy.rand()*( self.bfp[i] - self.pop2[i]) 
   self.v[i] = self.v[i] + self.c2*scipy.rand()*(self.bfg - self.pop2[i])
   # Atualiza posicao
   self.pop2[i] = self.pop2[i] + self.v[i]   
   self.ans2[i],self.pop2[i] = self.resolve_desafio(self.pop2[i])
   self.fit2[i] = self.avalia_aptidao2(self.ans2[i])
   self.bfp_fitness[i] = self.avalia_aptidao2(self.bfp_ans[i])
   self.bfg_fitness = self.avalia_aptidao2(self.bfg_ans)
   # Atualiza melhor posicao da particula
   if (self.fit2[i] > self.bfp_fitness[i]):
    self.bfp[i] = self.pop2[i]
    self.bfp_fitness[i] = self.fit2[i]
    self.bfp_ans[i] = self.ans2[i]
   # Atualiza melhor posicao global
   if (self.bfp_fitness[i] > self.bfg_fitness):
    self.bfg_fitness = self.bfp_fitness[i].copy()
    self.bfg = self.bfp[i].copy()
    self.bfg_ans = self.bfp_ans[i].copy()	

 def run(self):
   for i in scipy.arange(self.npop1):
    self.fit1[i] = self.avalia_aptidao1(self.ans1[i])	 
   self.Evolve_DE()
   self.HF1_Updt(self.ans1[self.fit1.argmax()],self.pop1[self.fit1.argmax()])   
   self.Evolve_PSO()
   self.HF2_Updt(self.bfg_ans,self.bfg)
   
class de:

 def __init__(self,fitness_func,npop = 10,pr = 0.7,beta = 2.5,debug=False):
  self.ns = npop
  self.beta = beta
  self.pr  = pr 
  self.debug = debug
  self.fitness_func = fitness_func
  self.fit = scipy.zeros((self.ns,1))
  self.pop = []
  # avalia fitness de toda populacao
  for i in scipy.arange(self.ns):
   self.fit[i],aux = self.avalia_aptidao(self.gera_individuo())
   #print i,self.fit[i]
   self.pop.append(aux.copy())
  self.pop = scipy.array(self.pop)
 
 def gera_individuo(self):
   return np.array([125*rand()+0.25 for i in range(Dim)]) 

 def avalia_aptidao(self,x):
  for i in range(x.shape[0]):
   if not 0.25 <= x[i] <= 125.25:
    x[i] = 125*rand()+0.25
  return (self.fitness_func(x),x)
  
 def run(self):  
  #prox_geracao = []
   
  for i in scipy.arange(self.ns):
   if self.debug: print "i = {0}".format(i)
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
    if self.debug: print "j (mutacao) = {0}".format(j)
    if self.debug: print "invalido flag = {0}".format(invalido)
   
   if self.debug: print "j (mutacao) = {0}".format(j)
   # trial vector a partir da mutacao de um alvo 
   u = self.pop[j[0]] + self.beta*(self.pop[j[1]] - self.pop[j[2]])
   if self.debug: print "u (target vector) = {0}".format(u)

   # gera por crossover solucao candidata
   c = self.pop[i].copy()  
   # seleciona indices para crossover
   # garantindo que ocorra crossover em
   # pelo menos uma vez
   j = random_integers(0,self.pop.shape[1]-1)
  
   for k in scipy.arange(self.pop.shape[1]):
    if (scipy.rand() < self.pr) or (k == j):
     c[k] = u[k]  

   c_fit,c = self.avalia_aptidao(c) 

   if self.debug: print "atual = {0}, fitness = {1}".format(self.pop[i],self.fit[i])
   if self.debug: print "candidato = {0}, fitness = {1}".format(c,c_fit)
    
   # leva para proxima geracao quem tiver melhor fitness
   if c_fit < self.fit[i]:
    self.pop[i] = c
    self.fit[i] = c_fit

  # avalia fitness da nova populacao
#  for i in scipy.arange(self.ns):
#   self.fit[i],self.pop[i] = self.avalia_aptidao(self.pop[i]) 

class pso:

 def __init__(self,fitness_func,npop = 20,w = 0.5,c1 = 2.01,c2 = 2.02,debug = False):
  self.debug = debug
  self.c1 = c1
  self.c2 = c2
  self.w = w
  self.ns = npop
  # gera pop inicial
  # centroides dos Kmax agrupamentos 
  self.pop = scipy.array([self.gera_individuo() for i in scipy.arange(self.ns)])
  self.fitness_func = fitness_func
  self.fit = scipy.zeros(self.ns)
  # avalia fitness de toda populacao
  for i in scipy.arange(self.ns):
   self.fit[i],self.pop[i] = self.avalia_aptidao(self.pop[i])  
   
  # inicializa velocidades iniciais
  self.v = scipy.array([self.gera_individuo() for i in scipy.arange(self.ns)])
  # guarda a melhor posicao de cada particula 
  self.bfp = scipy.copy(self.pop)
  self.bfp_fitness = scipy.copy(self.fit)
  # guarda a melhor posicao global
  self.bfg = self.pop[self.bfp_fitness.argmin()].copy()
  self.bfg_fitness = self.bfp_fitness.min().copy()

 def gera_individuo(self):
   return np.array([125*rand()+0.25 for i in range(Dim)])

 def avalia_aptidao(self,x): 
  for i in range(x.shape[0]):
   if not 0.25 <= x[i] <= 125.25:
    x[i] = 125*rand()+0.25
  return (self.fitness_func(x),x)
 
 def run(self):
 
  for i in scipy.arange(self.ns):
   # Atualiza velocidade
   self.v[i] = self.w*self.v[i] 
   self.v[i] = self.v[i] + self.c1*scipy.rand()*( self.bfp[i] - self.pop[i]) 
   self.v[i] = self.v[i] + self.c2*scipy.rand()*(self.bfg - self.pop[i])
   # Atualiza posicao
   self.pop[i] = self.pop[i] + self.v[i]
   
   self.fit[i],self.pop[i] = self.avalia_aptidao(self.pop[i])
   
   # Atualiza melhor posicao da particula
   if self.fit[i] < self.bfp_fitness[i]:
    self.bfp[i] = self.pop[i]
    self.bfp_fitness[i] = self.fit[i]
   if self.debug:
    print "self.fit[{0}] = {1} bfp_fitness = {2}".format(i,self.fit[i],self.bfp_fitness[i])
  # Atualiza melhor posicao global
   if  self.bfp_fitness[i] < self.bfg_fitness:
    self.bfg_fitness = self.bfp_fitness[i].copy()
    self.bfg = self.bfp[i].copy()

def f1(x):
 aux = 0
 for i in scipy.arange(x.shape[0]-1):
  aux = aux + 100*(x[i+1] - x[i]**2)**2+(x[i]-1)**2 
 return aux 

def f2(x):
 xx = x[0]**2+x[1]**2 
 aux = 1 + scipy.cos(12*scipy.sqrt(xx))
 aux = aux / (.5*xx+2) 
 return -aux

def f3(x):
  k1 = 0
  k2 = 0
  k3 = 1/float(x.shape[0])
  for i in range(x.shape[0]):
   k1 = k1 + x[i]**2
   k2 = k2 + scipy.cos(2*scipy.pi*x[i])
   
  return -20*scipy.exp(-0.2*scipy.sqrt(k3*k1))-scipy.exp(k3*k2)+20+scipy.exp(1)
 
