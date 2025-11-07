#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
import sys
import os
import getopt
import optimize
import pickle
import numpy as np
#from sklearn.preprocessing import scale
from functools import partial

mt = 1
fout = ""
dim = -1

try:                                
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["dim=","output="])
except getopt.GetoptError:           
 printf("Error getopt")                          
 sys.exit(2)          
 
for opt,arg in opts:
 if opt in ("-o","--output"):
  fout = arg
 elif opt == "--dim":
  dim = int(arg)
conf = [float(i) for i in args]
print(conf)
if fout == "" or len(conf) != 4 or dim <= 0:
 print("Error getopt") 
 sys.exit(2)

algo = "pso"
N,M = 200,1

Head = {'algo':algo,'conf':"npop = {0}, w = {1}, c1 = {2}, c2 = {3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':dim}
     
if __name__ == '__main__':
 optimize.set_dim(dim)

 with open(fout,"wb") as f:
  pickle.dump(Head,f)
  pickle.dump((N,M),f)

  for j in range(M):
   u = optimize.pso(optimize.f3,npop =conf[0],w = conf[1],c1 = conf[2],c2 = conf[3],delta = 0.8,alpha = 0.5)
   for i in range(N):
    u.run()
    print(j,i)
    print(u.bfg_fitness)
    #print(u.bfg_fitness,u.bfg)
    pickle.dump([i,u.bfg_fitness],f)
   
