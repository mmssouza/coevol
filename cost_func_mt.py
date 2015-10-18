
from multiprocessing import Pool,Manager
import numpy as np
import cPickle
import silhouette
import descritores as desc
from sklearn.preprocessing import scale

Y,cnt = [],[]

def DatasetLoad(d):
 with open(d+"classes.txt","r") as f:
  cl = cPickle.load(f)
  with open(d+"names.pkl","r") as f:
   nomes = cPickle.load(f)
   for k in nomes:
    cnt.append(desc.contour(d+k).c)
    Y.append(cl[k]) 

def fy(cc):
 s = l[0]
 return np.log(desc.bendenergy(cc,s)())

mgr = Manager()

l = mgr.list()

pool = Pool(processes=2) 

def cost_func(args):
 l.append(args)
 res = pool.map(fy,cnt)
 l.pop() 
 s = silhouette.silhouette(scale(np.array(res)),Y-1)
 # median absolute deviation (mad)
 return np.mean(np.abs(1.-s))
