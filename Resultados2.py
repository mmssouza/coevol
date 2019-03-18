#!/usr/bin/env python3

import sys
import pickle
import pylab

hall_of_fames = []
realizations = []
# coleta resultados
with open(sys.argv[1],'rb') as f:
 head_dict = pickle.load(f)
 dim = head_dict['dim']
 print(head_dict)
 N,M = pickle.load(f)
 print(N,M)
 for j in range(M):
   realizations.append([])
   hall_of_fames.append([])
   for k in range(N):
     r = pickle.load(f)
     realizations[j].append(r)
     stt = str("{:2} {:3.3} {:3.3} {:3.3} {:3.3}").format(r[0],r[1][0],r[1][1],r[1][2],r[1][3])
     print(stt+str().join([" {:2.3}".format(float(j)) for j in r[2]]))
   print("\n")
   for l in range(5):
    hf = pickle.load(f)
    hall_of_fames[j].append(hf)
    print(str().join(["{:3.3} ".format(float(i)) for i in hf]))
   print("\n")
 print("\n")
