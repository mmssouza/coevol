#!/usr/bin/python -u
import sys
import scipy
import numpy as np
from sklearn.preprocessing import scale
import cPickle
import silhouette
import descritores as desc
import optimize
import cost_func_mt

fd = open (sys.argv[1],'wb') 
optimize.set_dim(7)
sa = optimize.sim_ann(cost_func_mt.cost_func,0.125+ np.random.rand(optimize.Dim) + 25,180,0.95,10,5)
for j in range(3):
 for i in range(1000):
  sa.run()
   
fd.close()   

N,M = 1250,3
with open(sys.argv[1],"wb") as f:
 cPickle.dump((N,M),f)
 for j in range(M):
  sa = optimize.sim_ann(cost_func_mt.cost_func,0.125+ np.random.rand(optimize.Dim) + 25,180,0.95,10,5)
  for i in range(N):
   sa.run()
   print i,sa.fit
   print sa.hall_of_fame[0]
   cPickle.dump([i,sa.fit,sa.s,sa.hall_of_fame[0]],f) 
  
