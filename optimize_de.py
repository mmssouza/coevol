#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

import sys
import os
import getopt
import optimize 
import scipy
import pickle
import numpy as np
from functools import partial
 
fout = ""
dim = -1

try:                                
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["dim=","output="])
except getopt.GetoptError:           
 print("Error getopt")                          
 sys.exit(2)          
 
for opt,arg in opts:
 if opt in ("-o","--output"):
  fout = arg
 elif opt == "--dim":
  dim = int(arg) 

conf = [float(i) for i in args]
if fout == "" or len(conf) != 3 or dim <= 0:
 print("Error getopt") 
 sys.exit(2)

algo = "de"

N,M = 250,1

Head = {'algo':algo,'conf':" npop = {0}, pr = {1}, beta = {2}".format(conf[0],conf[1],conf[2]),'dim':dim}

if __name__ == '__main__':
 optimize.set_dim(dim)
   
 with open(fout,"wb",0) as f:
   pickle.dump(Head,f)
   pickle.dump((N,M),f)
   for j in range(M):
    v = optimize.de(optimize.f3,conf[0],conf[1],conf[2])
    for i in range(N):
     v.run()
     print(i,v.fit.min())
     print(v.pop[v.fit.argmin()])
     pickle.dump([i,v.fit.min()],f)
  
