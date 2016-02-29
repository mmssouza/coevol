#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import sys
import getopt
import optimize 
import scipy
import cPickle
from multiprocessing import Pool,Manager
import numpy as np
import metrics
import descritores as desc
from sklearn.preprocessing import scale
from functools import partial
 
mt = 1
dataset = ""
fout = ""
dim = -1

try:                                
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["mt=","dim=","output=","dataset="])
except getopt.GetoptError:           
 print "Error getopt"                          
 sys.exit(2)          
 
for opt,arg in opts:
 if opt == "--mt":
  mt = int(arg)
 elif opt in ("-o","--output"):
  fout = arg
 elif opt in ("-d","--dataset"):
  dataset = arg
 elif opt == "--dim":
  dim = int(arg) 

conf = [float(i) for i in args]
if dataset == "" or fout == "" or len(conf) != 3 or dim <= 0:
 print "Error getopt" 
 sys.exit(2)

algo = "de"
N,M = 300,15

Head = {'algo':algo,'conf':" npop = {0}, pr = {1}, beta = {2}".format(conf[0],conf[1],conf[2]),'dim':dim,"dataset":dataset}

def ff(cc,s):
  return np.log(desc.bendenergy(cc,s)())
     
if __name__ == '__main__':
 Y,cnt = [],[]
 with open(dataset+"/"+"classes.txt","r") as f:
  cl = cPickle.load(f)
  with open(dataset+"/"+"names.pkl","r") as f:
   nomes = cPickle.load(f)
   for k in nomes:
    cnt.append(desc.contour(dataset+"/"+k,method = 'octave').c)
    Y.append(cl[k])
	
 pool = Pool(processes=mt)
 
 def cost_func(args):   
  partial_ff = partial(ff, s = args)
  res = pool.map(partial_ff,cnt)
  s = metrics.silhouette(scale(np.array(res)),np.array(Y)-1)
  return np.median(np.abs(1.-s))

 optimize.set_dim(dim)

 with open(fout,"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  for j in range(M):
   v = optimize.de(cost_func,conf[0],conf[1],conf[2])
   for i in range(N):
    v.run()
    print i,v.fit.min()
    print v.pop[v.fit.argmin()]
    cPickle.dump([i,v.fit.min(),v.pop[v.fit.argmin()]],f)
   print "############################################"
