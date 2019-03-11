#!/usr/bin/env python3
# -*- coding: iso-8859-1 -*-
import sys
import pylab
import pickle
import descritores as desc
d = sys.argv[1]
dd = sys.argv[2]
f = open(d+"names.pkl","rb")
nomes = pickle.load(f)
f.close()
pylab.figure(1)
for fname in nomes:
 c = desc.contour_base(d+fname,nc=256,method = 'cv')
 pylab.scatter(pylab.array([i.real for i in c.c]),pylab.array([i.imag for i in c.c]),s = 5)
 print(fname)
 pylab.savefig(dd+fname)
 pylab.cla()
