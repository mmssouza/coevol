#!/usr/bin/python3


# Code source: Gael Varoqueux
# Modified for Documentation merge by Jaques Grobler
# License: BSD
import warnings
import numpy as np
from sklearn import neighbors,decomposition,model_selection,pipeline,metrics,naive_bayes
import sys
import pickle
from sklearn.preprocessing import scale
import descritores as desc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import some data to play with
warnings.simplefilter("ignore")

ss = np.array([0.559,2.3,74,4.74,1.8])
path = sys.argv[1]

with open(path+"classes.txt","rb") as f:
 with open(path+"names.pkl","rb") as g:
   cl = pickle.load(f)
   nomes = pickle.load(g)

db = {}
for im_file in nomes:
    nmbe = desc.bendenergy(path+im_file,ss)
    db[im_file] = np.hstack((cl[im_file],np.log(nmbe())))
    # nome das figuras
Y = np.array([db[i][0] for i in db.keys()]).astype(int)
X = np.array([db[i][1:] for i in db.keys()])
#s = float(Y.shape[0])
#priors = np.array([float(np.where(Y == i)[0].shape[0])/s for i in range(1,Y.max()+1)])

classifiers = ['nb','knn','lda','qda']


clf =     [pipeline.Pipeline([('lda', LinearDiscriminantAnalysis(n_components = 5)),('nb',naive_bayes.GaussianNB())]),
           pipeline.Pipeline([('lda', LinearDiscriminantAnalysis(n_components = 5)),('knn',neighbors.KNeighborsClassifier(n_neighbors = 3))]),
           pipeline.Pipeline([('lda', LinearDiscriminantAnalysis(n_components = 5))])]
		   #pipeline.Pipeline([('qda',qda.QDA())])]
it = model_selection.StratifiedShuffleSplit(n_splits = 5000,train_size = 0.7)
#it = model_selection.KFold(Y.size,n_folds = 3)
for c,cn in zip(clf,classifiers):
  res = model_selection.cross_val_score(c,scale(X),Y,cv = it,scoring = "accuracy")
  print(cn+': ',res.mean(),res.std())
