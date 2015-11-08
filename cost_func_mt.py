
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
 return np.log(desc.bendenergy(c,s[1:(int(s[0])+1)])())

mgr = Manager()

l = mgr.list()

pool = Pool(processes=4) 

def cost_func(args):
 l.append(args)
 res = pool.map(fy,cnt)
 l.pop() 
 s = silhouette.silhouette(scale(np.array(res)),np.array(Y)-1)
 #u = np.array([np.mean(np.abs(1. - s[np.array(Y) == i])) for i in range(1,max(Y)+1)])
 #return u.sum()
 # median absolute deviation (mad)
 return np.median(np.abs(1.-s))
