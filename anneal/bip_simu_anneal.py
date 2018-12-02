# WORKS

import dwave 
import neal
import numpy as np
from pprint import pprint
import dimod

A = np.array([-2, 6, -3, 4, 1])
c = np.array([3, 5, 6, 9, 10])
b = 2

k = b - sum(A)/2

var = len(A)
linear = {}
quadratic = {}
for i in range(var):
    mul = k*A[i] - 0.05*c[i]
    linear[i] = mul

for i in range(var-1):
    for j in range(i+1,var):
        mul = A[i]*A[j]*0.5
        quadratic[(i,j)] = mul

sampler = neal.SimulatedAnnealingSampler()
response = sampler.sample_ising(linear,quadratic)

for sample in response:
    pprint(sample)