# WORKING

from dimod.reference.samplers import ExactSolver
import dwavebinarycsp as dbc
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
import numpy as np
import dimod

A = np.array([-2, 6, -3, 4, 1])
c = np.array([3, 5, 6, 9, 10])
b = 2

lin = []
qua = []
linear = {}
quadratic = {}

for i in range(len(A)):
    lin.append( A[i]**2 - 2*b*A[i] + 0.05*c[i] )

for i in range(len(A)-1):
    for j in range(i+1,len(A)):
        qua.append( 2 * A[i] * A[j] )

scaling_factor = max(max(np.abs(lin)), max(np.abs(qua)))

for i in range(len(lin)):
    linear[i] = lin[i] / scaling_factor
n = 0
for i in range(len(A)-1):
    for j in range(i+1,len(A)):
        quadratic[(i,j)] = qua[n] / scaling_factor
        n += 1

bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
sampler = ExactSolver()
response = sampler.sample(bqm)

for sample,energy,num in response.data(['sample', 'energy', 'num_occurrences']):
    print(sample, energy, num)
