#!/usr/bin/python3 -u

import sys

import getopt
import pickle
import optimize
import numpy as np
from functools import partial
import time
from multiprocessing import shared_memory, Process
import numpy as np

def de_process(Niter,ff,npop,pr,beta,Ns,shm_name):

    de = optimize.de(ff,npop,pr,beta)

    shm = shared_memory.SharedMemory(name=shm_name)

    pool = np.ndarray((Ns,optimize.Dim), dtype=float, buffer=shm.buf)
    pool_pso = pool[int(Ns/2):]
    pool_de  = pool[0:int(Ns/2)]

    for i in range(Niter):
     de.run()
     min_fit= de.fit.min()
     print(f"de process:  {i:4d} {min_fit:2.3f}")
     #idx1 = np.argsort(de.fit)
     idx1 = np.random.permutation(npop)[0:int(Ns/2)]
     #pool_de[:] = de.pop[idx1][0:int(Ns/2)]
     pool_de[:] = de.pop[idx1]
     # Close the shared memor0y instance in the child process
     if i % 50 == 0:
      idx2 = np.random.permutation(npop)[0:int(Ns/2)]
      de.pop[idx2]= pool_pso
      for s,j in zip(de.pop[idx2],idx2):
       de.fit[j]= ff(s)

    shm.close()

def pso_process(Niter,ff,npop,w,c1,c2,Ns,shm_name):

    pso = optimize.pso(ff,npop,w,c1,c2)

    shm = shared_memory.SharedMemory(name=shm_name)

    pool = np.ndarray((Ns,optimize.Dim), dtype=float, buffer=shm.buf)
    pool_pso = pool[int(Ns/2):]
    pool_de  = pool[0:int(Ns/2)]

    for i in range(Niter):
     pso.run()
     min_fit = pso.fit.min()
     print(f"pso process: {i:12d} {min_fit:2.3f}")
     #idx1 = np.argsort(pso.fit)
     #pool_pso[:] = pso.pop[idx1][0:int(Ns/2)]
     idx1 = np.random.permutation(npop)[0:int(Ns/2)]
     pool_pso[:] = pso.pop[idx1]
     # Close the shared memory instance in the child process
     if i % 50 == 0:
      idx2 = np.random.permutation(npop)[0:int(Ns/2)]
      pso.pop[idx2]= pool_de
      for s,j in zip(pso.pop[idx2],idx2):
       pso.fit[j]= ff(s)

    shm.close()


if __name__ == "__main__":
# objective function dimension
    DIM = 10000

    optimize.set_dim(DIM)
# pool of solutions (must be even)
    Ns = 4
    rnd = np.random.default_rng().random(size=(Ns,DIM),dtype=float)
# shared memory for pool where optimizers share their best solutions
    shm = shared_memory.SharedMemory(create=True, size=Ns*DIM*rnd.itemsize)
    pool = np.ndarray((Ns,DIM),dtype=float,buffer=shm.buf)
    pool[:,:] = rnd

    # Create and start optimizers (child processes)
    p1 = Process(target=de_process, args=(2000,optimize.f3,100,0.4,0.9,Ns,shm.name))
    p2 = Process(target=pso_process, args=(2000,optimize.f3,100,0.05,2.3,2.3,Ns,shm.name))
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
