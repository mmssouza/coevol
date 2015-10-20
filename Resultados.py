#!/usr/bin/python

import sys
import cPickle
import pylab

with f as open(sys.argv[1],'r'):
  Head = cPickle.load(f)
  N,M = cPickle.load(f)
  index = []
  fitness = [[]]*M
  ans = [[]]*M
  for i in range(M)
   for j in range(N)
     aux = cPickle.load(f):
     if M == 0:
      index.append(aux[0])
     fitness.append(aux[1])
     ans.append(aux[2])
  for aux in Head.keys():
   print Head[aux]

  for i in range(M):
  pylab.plot(fitness[i])
  
