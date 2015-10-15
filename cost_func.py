
import numpy as np
import cPickle
import silhouette
import descritores as desc
from sklearn.preprocessing import scale

diretorio = "../leaves_png/"

with open(diretorio+"classes.txt","r") as f:
 cl = cPickle.load(f)
f.close()

with open("nomes_sampled.pkl","r") as f:
 nomes = cPickle.load(f)
f.close()

cnt = [desc.contour(diretorio+k).c for k in nomes]

Y = np.array([cl[k] for k in nomes])  

def cost_func(args):
 X = []
 for im_file in cl.keys(): 
  nmbe  = desc.bendenergy(diretorio+im_file,args)
  X.append(np.log(nmbe()))
 X = scale(np.array(X))
 
 s = silhouette.silhouette(X,Y-1)
# s = silhouette_score(X,labels) 
# return np.mean(s)
 return -np.median(s)
# return s.mean()


