#!/usr/bin/python -u
# -*- coding: utf-8 -*-
import sys
import optimize
import cost_func_mt
import scipy
import cPickle

algo = "de"
conf = [30,0.4,0.75]
dim = 7
dataset = "leaves_99_png"
N,M = 1000,3

Head = {'algo':"algo: "+algo,'conf':" npop = {0}, pr = {1}, beta = {2}".format(conf[0],conf[1],conf[2]),'dim':"n = {0}".format(dim),"dataset":"dataset: "+dataset}

depso.set_dim(dim)

cost_func_mt.DatasetLoad("../"+dataset+"/")

if __name__ == '__main__':
 with open(sys.argv[1],"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  for j in range(M):
   v = optimize.de(cost_func_mt.cost_func,conf[0],conf[1],conf[2])
   for i in range(N):
    v.run()
    print i,v.fit.min()
    print v.pop[v.fit.argmin()]
    cPickle.dump([i,v.fit.min(),v.pop[v.fit.argmin()]],f)
   print "############################################"
