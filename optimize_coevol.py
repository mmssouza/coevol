#!/usr/bin/python -u

import sys
import getopt
import cPickle
import optimize

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

if dataset == "" or fout == "" or len(conf) != 8 or dim <= 0:
 print "Error getopt" 
 sys.exit(2)

if mt:
 from cost_func_mt import cost_func,DatasetLoad
else:
 from cost_func import cost_func,DatasetLoad

algo = "coevol"
N,M = 1200,3

Head = {'algo':algo,'conf':"ns = {0}, de: (npop,pr,alpha) = ({1}, {2}, {3}), pso: (npop,w,c1,c1) = ({4},{5},{6},{7})".format(conf[0],conf[1],conf[2],conf[3],conf[4],conf[5],conf[6],conf[7]),'dim':dim,"dataset":dataset}

optimize.set_dim(dim)

DatasetLoad(dataset+"/")

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

