#!/usr/bin/python -u

import sys
import numpy as np
import cPickle
import optimize
import cost_func

N,M = 1000,1
optimize.set_dim(7)
cost_func.DatasetLoad("../leaves_99_png/")
with open(sys.argv[1],"wb") as f:
 cPickle.dump((N,M),f)
 for j in range(M):
  sa = optimize.sim_ann(cost_func.cost_func,0.125 + 125*np.random.rand(optimize.Dim),180,0.965,20,5)
  for i in range(N):
   sa.run()
   print i,sa.fit
   print sa.hall_of_fame[0]
   cPickle.dump([i,sa.fit,sa.s,sa.hall_of_fame[0]],f) 
  
