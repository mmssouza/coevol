#!/usr/bin/python -u

import sys
import cPickle
import optimize
import cost_func_mt 

algo = "coevol"
conf = (10,50,0.3,0.75,50,0.965,0.68,0.68)
dim = 5
dataset = "leaves_99_png"
N,M = 1000,1

Head = {'algo':"algo: "+algo,'conf':"ns = {0}, de: (npop,pr,alpha) = ({1}, {2}, {3}), pso: (npop,w,c1,c1) = ({4},{5},{6},{7})".format(conf[0],conf[1],conf[2],conf[3],conf[4],conf[5],conf[6],conf[7]),'dim':"n = {0}".format(dim),"dataset":"dataset: "+dataset}

optimize.set_dim(dim)

cost_func_mt.DatasetLoad("../"+dataset+"/")

with open(sys.argv[1],"wb") as f:
 cPickle.dump(Head,f)
 cPickle.dump((N,M),f)
 for j in range(M):
  w = optimize.coevol(cost_func_mt.cost_func,ns = conf[0],npop1 = conf[1],pr = conf[2],beta = conf[3],npop2 = conf[4],w = conf[5],c1 = conf[6],c2 = conf[7])
  for i in range(N):
   w.run()
   print i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.bfg_fitness,w.bfg_ans
   cPickle.dump([i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.pop1[w.fit1.argmax()]],f)
   cPickle.dump([i,w.bfg_fitness,w.bfg_ans,w.bfg],f)
  cPickle.dump(w.hall_of_fame1[0],f)
  cPickle.dump(w.hall_of_fame2[0],f)  
