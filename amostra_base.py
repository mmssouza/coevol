#!/usr/bin/python

import sys
import numpy as np
import cPickle
N = 3
diretorio = sys.argv[1]
cl = cPickle.load(open(diretorio+"classes.txt"))
classes = np.array(cl.values())
samples_list = []
for i in np.arange(1,classes.max()+1):
 idx1 = np.where(classes == i)
 idx2 = np.random.permutation(idx1[0].shape[0])
 idx3 = idx1[0][idx2[0:N]]
 samples_list = samples_list + [cl.keys()[j] for j in idx3]

cPickle.dump(samples_list,open(sys.argv[2],'wb'))
 
