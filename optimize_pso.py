#!/usr/bin/python -u
# -*- coding: utf-8 -*-
import sys
import getopt
import optimize
import cost_func_mt
import cPickle

mt = False
dataset = ""
fout = ""
dim = -1

try:                                
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["mt","dim =","output=","dataset="])
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

if dataset == "" or fout == "" or len(conf) != 4 or dim <= 0:
 print "Error getopt" 
 sys.exit(2)

if mt:
 from cost_func_mt import cost_func,DatasetLoad
else:
 from cost_func import cost_func,DatasetLoad

algo = "pso"
N,M = 1200,3

Head = {'algo':algo,'conf':"npop = {0}, w = {1}, c1 = {2}, c2 = {3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':dim,"dataset":dataset}

optimize.set_dim(dim)

cost_func_mt.DatasetLoad(dataset+"/")

if __name__ == '__main__':
 with open(sys.argv[1],"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  for j in range(M):
   u = optimize.pso(cost_func_mt.cost_func,npop =conf[0],w = conf[1],c1 = conf[2],c2 = conf[3])
   for i in range(N):
    u.run()
    print i,u.bfg_fitness
    print u.bfg
    cPickle.dump([i,u.bfg_fitness,u.bfg],f)
   print "############################################"
