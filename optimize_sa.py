#!/usr/bin/python -u

import sys
import getopt
import cPickle
import optimize
import pylab
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

algo = "sa"
N,M = 300,15

Head = {'algo':algo,'conf':"T0,alpha,P,L = {0},{1},{2},{3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':dim,"dataset":dataset}

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

 with open(fout,"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  for j in range(M):
   w = optimize.sim_ann(cost_func,0.125+pylab.rand(dim)*125.,conf[0],conf[1],conf[2],conf[3])
   for i in range(N):
    w.run()
    print i,w.fit
    cPickle.dump([i,w.fit,w.s],f)
   cPickle.dump(w.hall_of_fame[0],f)
  
