#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import sys
import getopt
import optimize 
import scipy
import cPickle
from multiprocessing import Pool
import numpy as np
import metrics
import descritores as desc
import amostra_base
from functools import partial
import time

mt = 1
dataset = ""
fout = ""
dim = -1
NS = 10

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
N,M = 100,15

Head = {'algo':algo,'conf':" npop = {0}, pr = {1}, beta = {2}".format(conf[0],conf[1],conf[2]),'dim':dim,"dataset":dataset,"NS":NS}

#def fc(fn,nc,oc):
# im = oc.imread(dataset+"/"+fn)
# s = oc.extract_longest_cont(im,nc)
# return np.array([complex(i[0],i[1]) for i in s])

#def fx(n,s):
# return desc.contour_base(dataset+k,method = "octave",nc = s).c  

def fy(n,s):
 tmp = desc.cd(dataset+n,method = 'octave',nc = s[3])
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
  #contours = [desc.contour_base(dataset+k,method = "octave",nc = args[3]).c for k in nomes]
  #contours = pool.map(partial(fx,s = args[3]),nomes) 
  #res = pool.map(partial(fy,s = args),contours)
  res = pool.map(partial(fy,s = args),nomes) 
  sys.stdout.write('. ')
  return np.median(1. - metrics.silhouette(np.array(res),np.array(Y)-1))
 
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
 oc.exit()
 
