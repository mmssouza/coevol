import sys
from multiprocessing import Pool,Manager
import numpy as np
import cPickle
import metrics
import descritores 
from sklearn.preprocessing import scale

Y,cl = [],{}
path = ""

def DatasetLoad(d):
 setattr(sys.modules[__name__],"path",d)
 with open(path+"classes.txt","r") as f:
  c = cPickle.load(f)
  for i,j in zip(c.keys(),c.values()):
   Y.append(int(j))
   cl[i] = j

def fy(im_file):
 s = l[0]
 tmp = descritores.cd(im_file)
 tmp_h = np.histogram(tmp,bins = int(s[0]),range = (s[1],s[2]))[0]
 tmp_h = tmp_h.astype(float)/tmp_h.sum()
 return tmp_h

mgr = Manager()

l = mgr.list()

pool = Pool(processes = 2) 

def cost_func(args):
 l.append(args)
 caux = [ path+i for i in cl.keys()]
 res = pool.map(fy,caux)
 l.pop() 
 return metrics.CS(np.array(res),np.array(Y)-1)
