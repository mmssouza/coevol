
import numpy as np
import cPickle
import silhouette
import descritores as desc
from sklearn.preprocessing import scale
diretorio = "../figs/"

f = open(diretorio+"classes.txt","r")
cl = cPickle.load(f)
f.close()
 
def cost_func(args):
 
 X = []
 Y = []
 for im_file in cl.keys(): 
  nmbe  = desc.bendenergy(diretorio+im_file,args)
  X.append(np.log(nmbe()))
  Y.append(cl[im_file])

 X = scale(np.array(X))
 Y = np.array(Y)

 s = silhouette.silhouette(X,Y-1)
# s = silhouette_score(X,labels) 
# return np.mean(s)
 return -np.median(s)
# return s.mean()


