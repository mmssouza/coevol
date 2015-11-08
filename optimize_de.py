#!/usr/bin/python -u
# -*- coding: utf-8 -*-
import sys
import getopt
import optimize
import scipy
import cPickle

mt = False
dataset = ""
fout = ""
dim = -1
try:                                
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["mt","dim=","output=","dataset="])
except getopt.GetoptError:           
 print "Error getopt"                          
 sys.exit(2)          
 
for opt,arg in opts:
 if opt == "--mt":
  mt = True
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

if mt:
 from cost_func_mt import cost_func,DatasetLoad
else:
 from cost_func import cost_func,DatasetLoad
 
algo = "de"
N,M = 1200,3

Head = {'algo':algo,'conf':" npop = {0}, pr = {1}, beta = {2}".format(conf[0],conf[1],conf[2]),'dim':dim,"dataset":dataset}

optimize.set_dim(dim)

DatasetLoad(dataset+"/")

if __name__ == '__main__':
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
