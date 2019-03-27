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

if dataset == "" or fout == "" or len(conf) != 4 or dim <= 0:
 print("Error getopt")
 sys.exit(2)

algo = "sa"
has_dump_file = False
if os.path.isfile("dump_optimize_sa2.pkl"):
 has_dump_file = True
 dump_fd = open("dump_optimize_sa2.pkl","rb")
 nn = pickle.load(dump_fd)
 mm = pickle.load(dump_fd)
else:
 nn = 0
 mm = 0

N,M = 200,5

Head = {'algo':algo,'conf':"T0,alpha,P,L = {0},{1},{2},{3}".format(conf[0],conf[1],conf[2],conf[3]),'dim':dim,"dataset":dataset}

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
    #print(k,cl[k],cnt)
  #nm = amostra_base.amostra(dataset,NS)
  #for i in nm:https://mail.google.com/mail/u/0/#search/Fl%C3%A1vio
  # Y.append(cl[i])
  # cnt.append(desc.contour(dataset+"/"+i).c)

 pool = Pool(processes=mt)

 def cost_func(args):
  partial_ff = partial(ff, s = args[0])
  res = pool.map(partial_ff,cnt)
  si = metrics.silhouette(scale(np.array(res)),np.array(Y)-1,distance = 'mahalanobis')
  db = metrics.db(scale(np.array(res)),np.array(Y)-1)
  ch = metrics.ch(scale(np.array(res)),np.array(Y)-1)
  w1,w2,w3 = args[1],args[2],args[3]
  fit = 3*w1*(np.std(si) + np.median(np.abs(1.-si)))+w2*db+50*w3/(ch+1e-12)

  return (np.mean(si),db,ch,fit)

 optimize.set_dim(dim)

 with open(fout,"ab",0) as f:
  if not has_dump_file:
   pickle.dump(Head,f)
   pickle.dump((N,M),f)
   #pickle.dump(w_list,f)
   tau1 = 1./np.sqrt(2*np.sqrt(N))
   tau2 = 1./np.sqrt(2*N)
   for j in range(mm,M):
    w = optimize.sim_ann(cost_func,conf[0],conf[1],conf[2],conf[3])
    w.tau1 = tau1
    w.tau2 = tau2
    for i in range(nn,N):
     w.run()
     dump_fd = open("dump_optimize_sa2.pkl","wb")
     pickle.dump(i+1,dump_fd)
     pickle.dump(j,dump_fd)
     dump_fd.close()
     ss = "{:2} ".format(i)
     ss = str().join([ss]+["{:,.3} ".format(float(k)) for k in w.s])
     ss = str().join([ss]+["{:,.3} ".format(float(w.fit[k])) for k in range(4)])
     print(ss)
     pickle.dump([i,w.fit,w.s],f)
    os.remove("dump_optimize_sa2.pkl")
    print()
    for hf in w.hall_of_fame:
     print(hf)
    print()
    pickle.dump(w.hall_of_fame[0],f)
    nn = 0
