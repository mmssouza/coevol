#!/usr/bin/python -u

import sys
import getopt
import cPickle
import optimize
import cost_func 
import pylab

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

if dataset == "" or fout == "" or len(conf) != 4 or dim <= 0:
 print "Error getopt" 
 sys.exit(2)

if mt:
 from cost_func_mt import cost_func,DatasetLoad
else:
 from cost_func import cost_func,DatasetLoad

algo = "sa"
N,M = 1200,3

Head = {'algo':algo,'conf':"T0,alpha,P,L = {0},{1},{2},{3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':dim,"dataset":dataset}

optimize.set_dim(dim)

cost_func.DatasetLoad(dataset+"/")

with open(fout,"wb") as f:
 cPickle.dump(Head,f)
 cPickle.dump((N,M),f)
 for j in range(M):
  w = optimize.sim_ann(cost_func.cost_func,0.125+pylab.rand(dim)*125.,conf[0],conf[1],conf[2],conf[3])
  for i in range(N):
   w.run()
   print i,w.fit
   cPickle.dump([i,w.fit,w.s],f)
  cPickle.dump(w.hall_of_fame[0],f)
  
