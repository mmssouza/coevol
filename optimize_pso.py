#!/usr/bin/python -u
# -*- coding: utf-8 -*-
import sys
import os
import getopt
import optimize
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

if dataset == "" or fout == "" or len(conf) != 4 or dim <= 0:
 print "Error getopt" 
 sys.exit(2)

algo = "pso"
has_dump_file = False
if os.path.isfile("dump_optimize_pso.pkl"):
 has_dump_file = True
 dump_fd = open("dump_optimize_pso.pkl","r")
 nn = cPickle.load(dump_fd)
 mm = cPickle.load(dump_fd)
else:
 nn = 0
 mm = 0  
N,M = 300,30

Head = {'algo':algo,'conf':"npop = {0}, w = {1}, c1 = {2}, c2 = {3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':dim,"dataset":dataset}

def ff(cc,s):
  return np.log(desc.bendenergy(cc,s)())
     
if __name__ == '__main__':
 Y,cnt = [],[]
 with open(dataset+"/"+"classes.txt","r") as f:
  cl = cPickle.load(f)
  with open(dataset+"/"+"names.pkl","r") as f:
   nomes = cPickle.load(f)
   for k in nomes:
    cnt.append(desc.contour(dataset+"/"+k).c)
    Y.append(cl[k])
	
 pool = Pool(processes=mt)
 
 def cost_func(args):   
  partial_ff = partial(ff, s = args)
  res = pool.map(partial_ff,cnt)
  s = metrics.silhouette(scale(np.array(res)),np.array(Y)-1)
  return np.median(np.abs(1.-s))

 optimize.set_dim(dim)
 
 with open(fout,"ab") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  if not has_dump_file:
   cPickle.dump(Head,f)
  for j in range(mm,M):
   u = optimize.pso(cost_func,npop =conf[0],w = conf[1],c1 = conf[2],c2 = conf[3])
   for i in range(nn,N):
    u.run()
    dump_fd = open("dump_optimize_pso.pkl","wb")
    cPickle.dump(i+1,dump_fd)
    cPickle.dump(j,dump_fd)
    dump_fd.close()
    print i,u.bfg_fitness
    print u.bfg
    cPickle.dump([i,u.bfg_fitness,u.bfg],f)
   os.remove("dump_pso.pkl")
   nn = 0
