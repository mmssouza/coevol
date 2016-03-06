#!/usr/bin/python

import sys
import descritores
import pylab
import cPickle
import metrics

db = cPickle.load(open(sys.argv[1]+"/classes.txt"))
#names = cPickle.load(open(sys.argv[1]+"/names.pkl"))
names = db.keys()
scales = pylab.loadtxt("mean_scale_de.pkl")
s1 = scales[0,0:11:2]
s2 = scales[1,0:11:2]
s3 = scales[2,0:11:2]
X1 = pylab.array([pylab.log(descritores.bendenergy(sys.argv[1]+f,s1)()) for f in names])
X2 = pylab.array([pylab.log(descritores.bendenergy(sys.argv[1]+f,s2)()) for f in names])
X3 = pylab.array([pylab.log(descritores.bendenergy(sys.argv[1]+f,s3)()) for f in names])

Y = pylab.array([db[n] for n in names])

print pylab.median(1.-metrics.silhouette(X1,Y-1))
print pylab.median(1.-metrics.silhouette(X2,Y-1))
print pylab.median(1.-metrics.silhouette(X3,Y-1))

