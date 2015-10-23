#!/usr/bin/python

import sys
import cPickle
import pylab

with open(sys.argv[1],'r') as f:
  try:
   while True:
     print cPickle.load(f)
  except:
    pass
