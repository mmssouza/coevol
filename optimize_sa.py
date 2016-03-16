#!/usr/bin/python -u

import sys
import getopt
import cPickle
import optimize
import pylab
from multiprocessing import Pool
import numpy as np
import metrics
import descritores as desc
import amostra_base
from functools import partial

mt = 1
dataset = ""
fout = ""
dim = -1
NS = 5
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
N,M = 300,5

Head = {'algo':algo,'conf':"T0,alpha,P,L = {0},{1},{2},{3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':dim,"dataset":dataset}

def ft(n,s):
 tmp = desc.TAS(n)
 tmp_h = np.histogram(tmp.sig,bins = int(s[0]),range = (s[1],s[2]))[0].astype(float)
 return tmp_h


def fy(n,s):
 tmp = desc.cd(n)
 tmp_h = np.histogram(tmp,bins = int(s[0]),range = (s[1],s[2]))[0].astype(float)
 #tmp_h = tmp_h.astype(float)/tmp_h.sum()
 return tmp_h

if __name__ == '__main__':
 Y,nomes = [],[]
 with open(dataset+"/"+"classes.txt","r") as f:
  cl = cPickle.load(f)
  n = amostra_base.amostra(dataset,NS)
  for k in n:
   Y.append(cl[k])
   nomes.append(k)
	
 pool = Pool(processes=mt)

 def cost_func(args):
  contours = [desc.contour_base(dataset+k,method = "octave",nc = args[3]).c for k in nomes]
  #contours = pool.map(partial(fx,s = args[3]),nomes) 
  #res = pool.map(partial(fy,s = args),contours)
  #res = [fy(c,args) for c in contours]
  res = [ft(c,args) for c in contours]
   
  sys.stdout.write('. ')
  return np.median(1. - metrics.silhouette(np.array(res),np.array(Y)-1))
 
 with open(fout,"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  for j in range(M):
   w = optimize.sim_ann(cost_func,conf[0],conf[1],conf[2],conf[3])
   for i in range(N):
    w.run()
    print i,w.fit
    cPickle.dump([i,w.fit,w.s],f)
   cPickle.dump(w.hall_of_fame[0],f)
  
