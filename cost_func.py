
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
	
def cost_func(args): 
 s = silhouette.silhouette(scale(np.log([desc.bendenergy(c,args)() for c in cnt])),np.array(Y)-1)
 # Median absolute deviation (MAD)
 return np.median(np.abs(1.-s))



