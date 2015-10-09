
import scipy
import numpy as np
from sklearn.preprocessing import scale
from multiprocessing import Pool,Manager
import cPickle
import silhouette
import descritores as desc
import optimize
 

def fy(cc):
 s = l[0]
 return np.log(desc.bendenergy(cc,s)())  

diretorio = "../figs/"
with open(diretorio+"classes.txt","r") as f:
  cl = cPickle.load(f)
cnt = [desc.contour(diretorio+k).c for k in cl.keys()]
Y = np.array([cl[k] for k in cl.keys()]) 

mgr = Manager()
l = mgr.list()
pool = Pool(processes=2)
 
def cost_func_mt(args):
  l.append(args)
  res = pool.map(fy,cnt)
  X = scale(np.array(res))
  l.pop() 
  s = silhouette.silhouette(X,Y-1)
  return -np.median(s)

def cost_func(args):
  X = []
  for c in cnt: 
   nmbe  = desc.bendenergy(c,args)
   X.append(np.log(nmbe()))
  
  X = scale(np.array(X))
  s = silhouette.silhouette(X,Y-1)
  return -np.median(s)
 
y = []  
w = optimize.coevol(cost_func_mt,ns = 10,npop1 = 50,pr = 0.45,beta = 0.4,npop2 = 220,w = 0.965,c1 = .38,c2 = .38)
for j in range(3):
  x1 = []
  for i in range(1250):
   w.run()
   print i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.bfg_fitness,w.bfg_ans
   for a,b,c in zip(w.sa.hall_of_fame,w.hall_of_fame1,w.hall_of_fame2):
    print a[0],b[0],c[0]
   x1.append([i,w.fit1.max(),w.ans1[w.fit1.argmax()],w.pop1[w.fit1.argmax()],w.bfg_fitness,w.bfg_ans])        
  y.append([x1,w.hall_of_fame1[0],w.hall_of_fame2[0],w.sa.hall_of_fame[0]])

with sys.argv[1] as f:
  cPickle.dump(y,f)

