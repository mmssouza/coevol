#!/usr/bin/python -u

import sys
import getopt
import cPickle
import optimize
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

if dataset == "" or fout == "" or len(conf) != 8 or dim <= 0:
 print "Error getopt" 
 sys.exit(2)

algo = "coevol"
N,M = 500,15

Head = {'algo':algo,'conf':"ns = {0}, de: (npop,pr,alpha) = ({1}, {2}, {3}), pso: (npop,w,c1,c1) = ({4},{5},{6},{7})".format(conf[0],conf[1],conf[2],conf[3],conf[4],conf[5],conf[6],conf[7]),'dim':dim,"dataset":dataset}

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
   w = optimize.coevol(cost_func,ns = conf[0],npop1 = conf[1],pr = conf[2],beta = conf[3],npop2 = conf[4],w = conf[5],c1 = conf[6],c2 = conf[7])
   for i in range(N):
    w.run()
    print i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.bfg_fitness,w.bfg_ans
    cPickle.dump([i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.pop1[w.fit1.argmax()]],f)
    cPickle.dump([i,w.bfg_fitness,w.bfg_ans,w.bfg],f)
   print w.hall_of_fame1[0]
   print w.hall_of_fame2[0]
   print "------------------------------------------------------"
   cPickle.dump(w.hall_of_fame1[0],f)
   cPickle.dump(w.hall_of_fame2[0],f)  

