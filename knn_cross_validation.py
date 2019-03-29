#!/usr/bin/python3 -u

import pylab
import warnings
import numpy as np
from sklearn import neighbors,model_selection,metrics
import sys
import pickle
from sklearn.preprocessing import scale
import descritores as desc

warnings.simplefilter("ignore")

ss = pylab.loadtxt(sys.argv[1])
path = sys.argv[2]
dim = ss.shape[1]-8
with open(path+"classes.txt","rb") as f:
 with open(path+"names.pkl","rb") as g:
   cl = pickle.load(f)
   nomes = pickle.load(g)

   clf = neighbors.KNeighborsClassifier(n_neighbors = 3)
   it = model_selection.RepeatedStratifiedKFold(n_splits = 5,n_repeats = 50)

   for s in ss:
    sigma = s[4:4+dim]
    SI,DB,CH = s[dim+4],s[dim+5],s[dim+6]

    db = {}
    for im_file in nomes:
      nmbe = desc.bendenergy(path+im_file,sigma)
      db[im_file] = np.hstack((cl[im_file],np.log(nmbe())))
    # nome das figuras

    Y = np.array([db[i][0] for i in db.keys()]).astype(int)
    X = scale(np.array([db[i][1:] for i in db.keys()]))
    res =  model_selection.cross_val_score(clf,X,Y,cv = it,scoring = "accuracy")
    st = str("{0} {1} {2} {3} {4} {5}").format(s[1],s[2],s[3],SI,DB,CH)

    print(" ".join([st]+["{:2.2f}".format(i) for i in sigma]+["{:0.2f} {:0.2f}".format(res.mean(),res.std())]))
    print
