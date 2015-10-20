#!/usr/bin/python -u
# -*- coding: utf-8 -*-
import sys
import optimize
import cost_func_mt
import cPickle

algo = "pso"
conf = [25,0.965,2.68,2.68]
dim = 5
dataset = "leaves_99_png"
N,M = 100,3

Head = {'algo':"algo: "+algo,'conf':"npop = {0}, w = {1}, c1 = {2}, c2 = {3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':"n = {0}".format(dim),"dataset":"dataset: "+dataset}

optimize.set_dim(dim)

cost_func_mt.DatasetLoad("../"+dataset+"/")

if __name__ == '__main__':
 with open(sys.argv[1],"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  for j in range(M):
   u = optimize.pso(cost_func_mt.cost_func,npop =conf[0],w = conf[1],c1 = conf[2],c2 = conf[3])
   for i in range(N):
    u.run()
    print i,u.bfg_fitness
    print u.bfg
    cPickle.dump([i,u.bfg_fitness,u.bfg],f)
   print "############################################"
