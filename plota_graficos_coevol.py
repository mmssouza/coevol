#!/usr/bin/python

import cPickle
import pylab

with open("coevol_teste.pkl","r") as f:
 head = cPickle.load(f)
 print head
 N,M = cPickle.load(f)
 print N,M
 l = []
 b = True
 while b:
  try:
   l.append(cPickle.load(f))
  except:
   b = False
 l = pylab.array(l).T.reshape((5*M,N)).T
 pylab.subplot(211) 
 for k in l[:,2*M:3*M].T:
  pylab.plot(l[:,0],k,"b")
 pylab.plot(l[:,0],pylab.mean(l[:,2*M:3*M].T,axis = 0),"r") 
 pylab.subplot(212)
 for k in l[:,4*M:5*M].T:
  pylab.plot(l[:,0],k,"g")
 pylab.plot(l[:,0],pylab.mean(l[:,4*M:5*M].T,axis = 0),"r")  
 
 pylab.show()