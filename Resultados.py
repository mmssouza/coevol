#!/usr/bin/env python3

import sys
import pickle
import pylab

hall_of_fames = []
realizations = []
# coleta resultados
with open(sys.argv[1],'rb') as f:
 head_dict = pickle.load(f)
 print(head_dict)
 N,M = pickle.load(f)
 print(N,M)
# weights = pickle.load(f)
 for i in range(len(weights)):
     realizations.append([])
     hall_of_fames.append([])
     for j in range(M):
         realizations[i].append([])
         hall_of_fames[i].append([])
         for k in range(N):
            realizations[i][j].append(pickle.load(f))
         hall_of_fames[i][j].append(pickle.load(f))
 for r in realizations:
 #for r,w in zip(realizations,weights):
     #print(str().join(["{:,.3} ".format(float(i)) for i in w]))
     print()
     for s in r:
         for ti in s:
          stt = str("{0} {1:,.3} {2:,.3} {3:,.3} {4:,.3} ").format(ti[0],ti[1][0],ti[1][1],ti[1][2],ti[1][3])
          print(stt+str().join(["{:,.3} ".format(float(j)) for j in ti[2]]))
         print()
 # formata e mostra resultados
 for hf in hall_of_fames:
     for h in hf:
        for x in h:
          print(str().join(["{:,.4} ".format(float(i)) for i in x]))
     print("\n")
