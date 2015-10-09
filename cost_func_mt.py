
import sys
from multiprocessing import Pool,Manager
import numpy as np
import cPickle
import silhouette
import descritores as desc
from sklearn.preprocessing import scale

def fy(cc):
 s = l[0]
 return np.log(desc.bendenergy(cc,s)())


diretorio = "../figs/"

with open(diretorio+"classes.txt","r") as f:
 cl = cPickle.load(f)

#with open("nomes_sampled.pkl","r") as f:
# nomes = cPickle.load(f)

cnt = [desc.contour(diretorio+k).c for k in cl.keys()]

Y = np.array([cl[k] for k in cl.keys()]) 

mgr = Manager()

l = mgr.list()

pool = Pool(processes=2) 

def cost_func(args):
 #sys.stdout.write(".")
 l.append(args)
 res = pool.map(fy,cnt)
 X = scale(np.array(res))
 l.pop() 
 s = silhouette.silhouette(X,Y-1)
# median absolute deviation (mad)
 return -np.median(s)
# mean square error (mse)
 #return ((1.-s)**2).mean()
# return s.mean()


