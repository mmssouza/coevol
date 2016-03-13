#!/usr/bin/python

import sys
import descritores
import pylab
import cPickle
import metrics

db = cPickle.load(open(sys.argv[1]+"/classes.txt"))
#names = cPickle.load(open(sys.argv[1]+"/names.pkl"))
names = db.keys()
scales = pylab.loadtxt(sys.argv[2])

X = [pylab.vstack(([db[f] for f in names],pylab.array([pylab.log(descritores.bendenergy(sys.argv[1]+f,s)()) for f in names]).T)).T for s in scales]

for x in X:
 s = metrics.silhouette(x[:,1:],x[:,0].astype(int)-1)
 print pylab.mean(s),pylab.std(s)

for n,x in zip(['nmbe_pso.pkl','nmbe_de.pkl','nmbe_sa.pkl'],X): 
 with open(n,"wb") as f:
  cPickle.dump(dict(zip(names,x)),f)


