#!/usr/bin/python3 -u

import sys
import getopt
import pickle
import optimize
import numpy as np
from functools import partial
import time

fout = ""
dim = -1
try:                                
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["dim=","output="])
except getopt.GetoptError:           
 print("Error getopt") 
 sys.exit(2)          
 
for opt,arg in opts:
 if opt in ("-o","--output"):
  fout = arg  
 elif opt == "--dim":
  dim = int(arg) 
  
conf = [float(i) for i in args]

if fout == "" or len(conf) != 8 or dim <= 0:
 print("Error getopt") 
 sys.exit(2)

algo = "coevol"
N,M = 455,1

Head = {'algo':algo,'conf':"ns1 = {0}, ns2 = {1}, de1: (npop,pr,beta) = ({2}, {3}, {4}), de2: (npop,pr,beta) = ({5},{6},{7})".format(conf[0],conf[1],conf[2],conf[3],conf[4],conf[5],conf[6],conf[7]),'dim':dim}
     
if __name__ == '__main__':
 optimize.set_dim(dim)

 with open(fout,"wb") as f:
  pickle.dump(Head,f)
  pickle.dump((N,M),f)
  for j in range(M):
   w = optimize.coevol(optimize.f3,ns1 = int(conf[0]),ns2 = int(conf[1]),npop1 = int(conf[2]),pr1 = conf[3],beta1 = conf[4],npop2 = int(conf[5]),pr2 = conf[6],beta2 = conf[7])
   for i in range(N):
    w.run()
    print (j,i)
    print ("de1 ",w.p1.fit.min(),w.ff(w.p1.pop[w.p1.fit.argmin()]))
	#print "de2 ",w.p2.fit.min(),w.p1.pop[w.p2.fit.argmin()]
    print ("de2 ",w.p2.fit.min(),w.ff(w.p2.pop[w.p2.fit.argmin()]))
	#time.sleep(0.2)
	#print
    #pickle.dump([i,w.p1.fit.min(),w.ff(w.p1.pop[w.p1.fit.argmin()]),w.p2.bfg_fitness,w.ff(w.p2.bfg)],f)
	#pickle.dump([i,w.p1.fit.min(),w.ff(w.p1.pop[w.p1.fit.argmin()]),w.p2.fit.min(),w.ff(w.p2.pop[w.p2.fit.argmin()])],f)
   #print("------------------------------------------------------")
   #print( w.hall_of_fame1[0])
   #print( w.hall_of_fame2[0])
   #print( "------------------------------------------------------")
   #pickle.dump(w.hall_of_fame1[0],f)
   #pickle.dump(w.hall_of_fame2[0],f)  

