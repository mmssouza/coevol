#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
import sys
import pylab
import cPickle
import descritores as desc
d = sys.argv[1]
f = open(d+"names.pkl","r")
nomes = cPickle.load(f)
f.close()
pylab.figure(1)
for fname in nomes:
 c = desc.contour_base(d+fname)
 pylab.scatter(pylab.array([i.real for i in c]),pylab.array([i.imag for i in c]),s = 5)
 print fname
 pylab.savefig(fname)
 pylab.cla()