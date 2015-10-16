#!/usr/bin/python -u

import sys
import scipy
import sys
import numpy as np
from sklearn.preprocessing import scale
from multiprocessing import Pool,Manager
import cPickle
import silhouette
import descritores as desc
import optimize
import cost_func_mt 

N,M = 1250,3
optimize.set_dim(7)
with open(sys.argv[1],"wb") as f:
 cPickle.dump((N,M),f)
 for j in range(M):
  w = optimize.coevol(cost_func_mt.cost_func,ns = 50,npop1 = 60,pr = 0.4,beta = 0.65,npop2 = 300,w = 0.965,c1 = .168,c2 = .168)
  for i in range(N):
   w.run()
   print i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.bfg_fitness,w.bfg_ans
   cPickle.dump([i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.pop1[w.fit1.argmax()]],f)
   cPickle.dump([i,w.bfg_fitness,w.bfg_ans,w.bfg],f)
  cPickle.dump(w.hall_of_fame1[0],f)
  cPickle.dump(w.hall_of_fame2[0],f)  

