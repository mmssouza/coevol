#!/usr/bin/python -u

import sys
import getopt
import cPickle
import optimize
from multiprocessing import Pool
import numpy as np
import descritores as desc
import amostra_base
from functools import partial
import rpyc

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True  

mt = 1
dataset = ""
fout = ""
dim = -1
NS = 5
Nc = 128

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

#CalcTas = rpyc.async(conn1.root.CalcTas)  

def Eval_Contours(nomes,mt,nc):
 conn1 = rpyc.connect("10.0.0.20", 18871)
 bgsrv1 = rpyc.BgServingThread(conn1)  
 
 Nobj = len(nomes)
# Gera as particoes (Npart listas de indices das linhas de data1)
 limits_hi= np.linspace(Nobj/mt,Nobj,mt).astype(int)
 limits_lo = np.hstack((0,limits_hi[0:limits_hi.shape[0]-1]))
 idx =[np.arange(lo,hi) for lo,hi in zip(limits_lo,limits_hi)]

 l = []
 for j in range(mt):
  l.append([])
  for n in np.array(nomes)[idx[j]]:
   l[j].append(str(n))

 evc_l = [conn1.root.EvalContours(l[j]) for j in range(mt)]

 for ev in evc_l:
   ev.set_nc(nc)
   ev.active()

 busy = True 
		  
 while busy:
   busy = False
   for evc in evc_l:
     busy = busy or evc.is_busy()
   
 cont = []
 for evc in evc_l:
  cont = cont + rpyc.utils.classic.obtain(evc.contours)  

 for evc in evc_l:
  evc.stop()  
 
 bgsrv1.stop()
 conn1.close()
 return cont

def ft(n,s):
 tmp = desc.TAS(n,method = 'octave').sig
 tmp_h = np.histogram(tmp,bins = int(s[0]),range = (s[1],s[2]))[0].astype(float)
 return tmp_h


def fy(n,s):
 tmp = desc.cd(n)
 tmp_h = np.histogram(tmp,bins = int(s[0]),range = (s[1],s[2]))[0].astype(float)
 #tmp_h = tmp_h.astype(float)/tmp_h.sum()
 return tmp_h

if __name__ == '__main__':
 import pdist_mt

 Y,nomes = [],[]
 with open(dataset+"/"+"classes.txt","r") as f:
  cl = cPickle.load(f)
  n = amostra_base.amostra(dataset,NS)
  for k in n:
   Y.append(cl[k])
   nomes.append(k)
 contours = Eval_Contours(nomes,mt,Nc)	
 pool = Pool(processes=mt)
 
 def cost_func(args):   
  #res2 = CalcTas(contours[len(contours)/2:],args)
  res = pool.map(partial(ft,s = args),contours)
  #res = list(res1)+list(res2.value)
  cost = float(np.median(1. - pdist_mt.silhouette(np.array(res),np.array(Y)-1)))
  
  sys.stdout.write('.')
  return cost
 
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
 