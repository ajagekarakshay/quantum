# WRONG MODEL - USE ISING

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
   # linear[i] = mul
    linear[(i,i)] = mul

for i in range(var-1):
    for j in range(i+1,var):
        mul = A[i]*A[j]*0.5
    #    quadratic[(i,j)] = mul
        quadratic[((i,j), (i,j))] = mul
        
bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY) 
sampler = neal.SimulatedAnnealingSampler()
#sampler = dimod.ExactSolver()
response = sampler.sample(bqm)

for a,b,c in response.data(['sample', 'energy', 'num_occurrences']):
    print(a,b,c)

