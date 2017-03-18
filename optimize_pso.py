#!/usr/bin/python -u
# -*- coding: utf-8 -*-
import sys
import os
import getopt
import optimize
import cPickle
import numpy as np
from sklearn.preprocessing import scale
from functools import partial

mt = 1
dataset = ""
fout = ""
dim = -1

try:                                
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["dim=","output="])
except getopt.GetoptError:           
 print "Error getopt"                          
 sys.exit(2)          
 
for opt,arg in opts:
 if opt in ("-o","--output"):
  fout = arg
 elif opt == "--dim":
  dim = int(arg)
conf = [float(i) for i in args]

if fout == "" or len(conf) != 6 or dim <= 0:
 print "Error getopt" 
 sys.exit(2)

algo = "pso"
N,M = 100,20

Head = {'algo':algo,'conf':"npop = {0}, w = {1}, c1 = {2}, c2 = {3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':dim,"dataset":dataset}
     
if __name__ == '__main__':
 optimize.set_dim(dim)

 with open(fout,"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)

  for j in range(M):
   u = optimize.pso(optimize.f3,npop =conf[0],w = conf[1],c1 = conf[2],c2 = conf[3],delta = 0.8,alpha = 0.5)
   for i in range(N):
    u.run()
    print j,i
    print u.bfg_fitness,u.bfg
    cPickle.dump([i,u.bfg_fitness],f)
   print "------------------------------------------------------"
   