#!/usr/bin/python -u

import sys
import cPickle
import optimize
import cost_func 

N,M = 1000,1
optimize.set_dim(7)
cost_func.DatasetLoad("../leaves_99_png/")
with open(sys.argv[1],"wb") as f:
 cPickle.dump((N,M),f)
 for j in range(M):
  w = optimize.coevol(cost_func.cost_func,ns = 15,npop1 = 30,pr = 0.4,beta = 0.75,npop2 = 30,w = 0.965,c1 = 2.68,c2 = 2.68)
  for i in range(N):
   w.run()
   print i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.bfg_fitness,w.bfg_ans
   cPickle.dump([i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.pop1[w.fit1.argmax()]],f)
   cPickle.dump([i,w.bfg_fitness,w.bfg_ans,w.bfg],f)
  cPickle.dump(w.hall_of_fame1[0],f)
  cPickle.dump(w.hall_of_fame2[0],f)  

