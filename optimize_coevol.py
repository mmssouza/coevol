#!/usr/bin/python -u

import sys
import getopt
import cPickle
import optimize
from multiprocessing import Pool,Manager
import numpy as np
import metrics
import descritores
import amostra_base
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

if dataset == "" or fout == "" or len(conf) != 8:
 print "Error getopt" 
 sys.exit(2)

algo = "coevol"
N,M = 100,1

Head = {'algo':algo,'conf':"ns = {0}, de: (npop,pr,alpha) = ({1}, {2}, {3}), pso: (npop,w,c1,c1) = ({4},{5},{6},{7})".format(int(conf[0]),int(conf[1]),conf[2],conf[3],int(conf[4]),conf[5],conf[6],conf[7]),'dim':dim,"dataset":dataset}

def fy(im_file,s):
 tmp = descritores.cd(im_file,method = 'octave',nc = int(s[3]))
 tmp_h = np.histogram(tmp,bins = int(s[0]),range = (s[1],s[2]))[0]
 #tmp_h = tmp_h.astype(float)/tmp_h.sum()
 return tmp_h

pool = Pool(processes = mt)      

if __name__ == '__main__':
 Y,cl,names = [],[],[]
 with open(dataset+"/"+"classes.txt","r") as f:
  cl = cPickle.load(f)
#  with open(dataset+"/"+"names.pkl",'r') as g:
#   for n in cPickle.load(g):
#    names.append(n)
  for n in amostra_base.amostra(dataset, N=5):
    names.append(n) 
  for k in names:
    Y.append(int(cl[k]))
	
 def cost_func(args):
  ff = partial(fy,s = args)
  caux = [dataset+i for i in names]
  res = pool.map(ff,caux)
  return metrics.CS(np.array(res),np.array(Y)-1)
 
 #optimize.set_dim(dim)

 with open(fout,"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  for j in range(M):
   w = optimize.coevol(cost_func,ns = int(conf[0]),npop1 = int(conf[1]),pr = conf[2],beta = conf[3],npop2 = int(conf[4]),w = conf[5],c1 = conf[6],c2 = conf[7])
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

