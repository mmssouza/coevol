#!/usr/bin/env python3
import numpy as np
import pylab
from scipy import spatial
from functools import reduce
import sys

def is_pareto(costs, maximise=False):
    """
    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

aux = pylab.loadtxt(sys.argv[1])[:,]
dim = len(aux[0]) - 8 
x = aux[:,dim+4:dim+7].copy()
x[:,0] = -x[:,0]
x[:,2] = -x[:,2]

for p in aux[is_pareto(x)]:
 print(str().join(["{:3.3} ".format(float(j)) for j in p]))

