#!/usr/bin/python3 -u

import sys

import getopt
import pickle
import optimize
import numpy as np
from functools import partial
import time
from multiprocessing import shared_memory,Lock,Process,Event
import numpy as np
import yaml

def de_p1(Niter,ff,conf,Ns,shm_name,lock,ev):
    npop = conf['npop']
    pr = conf['pr']
    beta = conf['beta']
    epoch = conf['epoch']

    de = optimize.de(ff,npop,pr,beta)

    shm = shared_memory.SharedMemory(name=shm_name)

    pool = np.ndarray((Ns,optimize.Dim), dtype=float, buffer=shm.buf)
    pool_pso = pool[int(Ns/2):]
    pool_de  = pool[0:int(Ns/2)]

    for i in range(1,Niter):

     ev[0].clear()

     de.run()

     ev[0].set()
     ev[1].wait()

     min_fit= de.fit.min()

     print(f" de: {i:4d} {min_fit:2.3f} ")

     idx1 = np.random.permutation(npop)[0:int(Ns/2)]

     with lock:
      pool_de[:] = de.pop[idx1]

      if i % epoch == 0:
       idx2 = np.random.permutation(npop)[0:int(Ns/2)]
       de.pop[idx2]= pool_pso

       for s,j in zip(de.pop[idx2],idx2):
        de.fit[j]= ff(s)

    shm.close()

def pso_p1(Niter,ff,conf,Ns,shm_name,lock,ev):

    npop = conf['npop']
    w = conf['w']
    c1 = conf['c1']
    c2 = conf['c2']
    epoch = conf['epoch']

    pso = optimize.pso(ff,npop,w,c1,c2)

    shm = shared_memory.SharedMemory(name=shm_name)

    pool = np.ndarray((Ns,optimize.Dim), dtype=float, buffer=shm.buf)
    pool_pso = pool[int(Ns/2):]
    pool_de  = pool[0:int(Ns/2)]

    for i in range(1,Niter):

     ev[1].clear()

     pso.run()

     ev[1].set()
     ev[0].wait()

     min_fit = pso.fit.min()
     print(f" pso: {i:4d} {min_fit:2.3f} ")

     idx1 = np.random.permutation(npop)[0:int(Ns/2)]

     with lock:
      pool_pso[:] = pso.pop[idx1]

      if i % epoch == 0:
       idx2 = np.random.permutation(npop)[0:int(Ns/2)]
       pso.pop[idx2]= pool_de

       for s,j in zip(pso.pop[idx2],idx2):
        pso.fit[j]= ff(s)

    shm.close()


if __name__ == "__main__":
# objective function dimension
    DIM = 1000

    optimize.set_dim(DIM)

    # parse configurations
    with open(sys.argv[1],'r') as f:
        conf = list(yaml.safe_load_all(f))

    Niter = conf[0]['Niter']
    Ns = conf[0]['Ns']
    de_conf = conf[1]
    pso_conf = conf[2]
    print(Niter,Ns)
    print(de_conf)
    print(pso_conf)
# pool of solutions (must be even)
    rnd = np.random.default_rng().random(size=(Ns,DIM),dtype=float)

# shared memory for pool where optimizers share their best solutions
    shm = shared_memory.SharedMemory(create=True, size=Ns*DIM*rnd.itemsize)
    pool = np.ndarray((Ns,DIM),dtype=float,buffer=shm.buf)
    pool[:,:] = rnd

    lock = Lock()
    ev = [Event(),Event()]

    # Create and start optimizers (child processes)
    p1 = Process(target=de_p1, args=(Niter,optimize.f3,de_conf,Ns,shm.name,lock,ev))
    p2 = Process(target=pso_p1, args=(Niter,optimize.f3,pso_conf,Ns,shm.name,lock,ev))

    p1.start()
    p2.start()

    p1.join()  # Wait for child processes to finish
    p2.join()

# print pool of soluctions and objective evaluated
    for pp in pool:
     print(pp,optimize.f3(pp))

    # Clean up: close and unlink the shared memory block
    shm.close()
    shm.unlink()
