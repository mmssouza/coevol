#!/usr/bin/python -u

import sys
import getopt
import cPickle
import optimize
import numpy as np
from functools import partial
import time

fout = ""
dim = -1
try:                                
 opts,args = getopt.getopt(sys.argv[1:], "o:d:", ["dim=","output="])
except getopt.GetoptError:           
 print "Error getopt"                          
 sys.exit(2)          
 
for opt,arg in opts:
 if opt in ("-o","--output"):
  fout = arg  
 elif opt == "--dim":
  dim = int(arg) 
  
conf = [float(i) for i in args]

if fout == "" or len(conf) != 11 or dim <= 0:
 print "Error getopt" 
 sys.exit(2)

algo = "coevol"
N,M = 55,10

Head = {'algo':algo,'conf':"ns1 = {0}, ns2 = {1}, de: (npop,pr,beta) = ({2}, {3}, {4}), pso: (npop,w,c1,c1,delta,alpha) = ({5},{6},{7},{8},{9},{10})".format(conf[0],conf[1],conf[2],conf[3],conf[4],conf[5],conf[6],conf[7],conf[8],conf[9],conf[10]),'dim':dim}
     
if __name__ == '__main__':
 optimize.set_dim(dim)

 with open(fout,"wb") as f:
  cPickle.dump(Head,f)
  cPickle.dump((N,M),f)
  for j in range(M):
   w = optimize.coevol(optimize.f3,ns1 = int(conf[0]),ns2 = int(conf[1]),npop1 = int(conf[2]),pr = conf[3],beta = conf[4],npop2 = int(conf[5]),w = conf[6],c1 = conf[7],c2 = conf[8],delta = conf[9],alpha = conf[10])
   for i in range(N):
	w.run()
	print j,i
	print "de ",w.p1.fit.min(),w.p1.pop[w.p1.fit.argmin()]
	#print "de2 ",w.p2.fit.min(),w.p1.pop[w.p2.fit.argmin()]
	print "pso ",w.p2.bfg_fitness,w.p2.bfg
	#time.sleep(0.2)
	#print
	cPickle.dump([i,w.p1.fit.min(),w.ff(w.p1.pop[w.p1.fit.argmin()]),w.p2.bfg_fitness,w.ff(w.p2.bfg)],f)
	#cPickle.dump([i,w.p1.fit.min(),w.ff(w.p1.pop[w.p1.fit.argmin()]),w.p2.fit.min(),w.ff(w.p2.pop[w.p2.fit.argmin()])],f)
   print "------------------------------------------------------"
   print w.hall_of_fame1[0]
   print w.hall_of_fame2[0]
   print "------------------------------------------------------"
   #cPickle.dump(w.hall_of_fame1[0],f)
   #cPickle.dump(w.hall_of_fame2[0],f)  

