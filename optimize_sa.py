#!/usr/bin/python3 -u

import sys
import os
import getopt
import pickle
import optimize
from multiprocessing import Pool
import numpy as np
import metrics
import descritores as desc
from sklearn.preprocessing import scale
from functools import partial

mt = 1
dataset = ""
fout = ""
dim = -1
NS = 2

try:
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["mt=","dim=","output=","dataset="])
except getopt.GetoptError:
 print("Error getopt")
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

if dataset == "" or fout == "" or len(conf) != 7 or dim <= 0:
 print("Error getopt")
 sys.exit(2)

algo = "sa"
#has_dump_file = False
#if os.path.isfile("dump_optimize_sa2.pkl"):
# has_dump_file = True
# dump_fd = open("dump_optimize_sa2.pkl","rb")
# nn = pickle.load(dump_fd)
# mm = pickle.load(dump_fd)
#else:
# nn = 0
# mm = 0

N,M = 350,3

Head = {'algo':algo,'conf':"T0,alpha,P,L,w0,w1,w2 = {0},{1},{2},{3},{4},{5},{6}".format(conf[0],conf[1],conf[2],conf[3],conf[4],conf[5],conf[6]),'dim':dim,"dataset":dataset}

weights = (float(conf[4]),float(conf[5]),float(conf[6]))

def ff(cc,s):
  return np.log(desc.bendenergy(cc,s)())

if __name__ == '__main__':
 Y,cnt = [],[]
 with open(dataset+"/"+"classes.txt","rb") as f:
  cl = pickle.load(f)
  with open(dataset+"/"+"names.pkl","rb") as f:
   nomes = pickle.load(f)
   for k in nomes:
    cnt.append(desc.contour(dataset+"/"+k,method='cv').c)
    Y.append(cl[k])

 pool = Pool(processes=mt)

 def cost_func(args):
  partial_ff = partial(ff, s = args[0])
  res = pool.map(partial_ff,cnt)
  si = metrics.silhouette(scale(np.array(res)),np.array(Y)-1)
  db = metrics.db(scale(np.array(res)),np.array(Y)-1)
  ch = metrics.ch(scale(np.array(res)),np.array(Y)-1)
  w1,w2,w3 = float(args[1]),float(args[2]),float(args[3])
  fit = (np.std(si)+np.mean(1.-si))*db/(ch+1e-12)

  return (np.mean(si),db,ch,fit)

 optimize.set_dim(dim)

 with open(fout,"ab",0) as f:
   pickle.dump(Head,f)
   pickle.dump((N,M),f)
   #pickle.dump(w_list,f)
   tau1 = 1./np.sqrt(2*np.sqrt(N))
   tau2 = 1./np.sqrt(2*N)
   for j in range(M):
    w = optimize.sim_ann(cost_func,conf[0],conf[1],conf[2],conf[3],weights)
    w.tau1 = tau1
    w.tau2 = tau2
    for i in range(N):
     w.run()
     ss = "{:} {:.2f} {:.2f} {:.2f} ".format(i,w.w1,w.w2,w.w3)
     ss = str().join([ss]+["{:,.3} ".format(float(k)) for k in w.s])
     ss = str().join([ss]+["{:,.3} ".format(float(w.fit[k])) for k in range(4)])
     print(ss)
     pickle.dump([i,w.fit,w.s],f)

    print()
    for hf in w.hall_of_fame:
     print(hf)
    print()
    pickle.dump(w.hall_of_fame[0],f)
