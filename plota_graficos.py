#!/usr/bin/python3

import pickle
import pylab
import sys

with open(sys.argv[1],"rb") as f:
 head = pickle.load(f)
 print(head)
 N,M = pickle.load(f)
 print(N,M)
 l = []
 b = True
 while b:
  try:
   l.append(pickle.load(f))
  except:
   b = False
 l = pylab.array(l)
 pylab.subplot(211) 
 pylab.plot(l[:,0],l[:,1],"b")
 pylab.show()
