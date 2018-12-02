import numpy as np
import dwave 
import neal
from pprint import pprint
import dimod
from dimod.reference.samplers import ExactSolver
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite, VirtualGraphComposite
from dwave.system.samplers import DWaveSampler
from time import time
import sys


fopen = open(sys.argv[1],'r')
data = fopen.readlines()
fopen.close()


def data_parser(data):
    N = int(data[0].rstrip())
    A = []
    B = []
    for i in range(2,2+N):
        a_line = data[i].rstrip().split(' ')
        a_line = [x for x in a_line if x!='']
        b_line = data[i+N+1].rstrip().split(' ')
        b_line = [x for x in b_line if x!='']
        A.append(np.array(a_line, dtype=float))
        B.append(np.array(b_line, dtype=float))
    return N, np.matrix(np.array(A)), np.matrix(np.array(B))
        
N, A, B = data_parser(data)

def prob_hamiltonian(N, A, B):
    # MultiPLIER ADDED AS 0.01
    M = A*B
    lin = {}
    qua = {}
    test = {}
    # Objective function
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    mul = A[i,j] * B[k,l] * 0.0001
                    if i==j and k==l: # Diferrence between quadratic and linear
                        try:
                            lin[(i,k,j,l)].append(mul)
                        except KeyError:
                            lin[(i,k,j,l)] = [mul]
                    else:  
                        try:
                            qua[(i,k,j,l)].append(mul)
                        except KeyError:
                            qua[(i,k,j,l)] = [mul]
    return lin, qua

lin, qua = prob_hamiltonian(N, A, B) 

def constraint_hamiltonian(lin, qua):
    # Ith constraint
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if j == k:
                    try:
                        lin[(i,j,i,k)].append(1)
                    except:
                        print("Something's wrong")
                else:
                    try:
                        qua[(i,j,i,k)].append(1)
                    except:
                        print("Something's wrong")
    # Jth constraint
    for j in range(N):
        for i in range(N):
            for k in range(N):
                if i == k:
                    try:
                        lin[(i,j,k,j)].append(1)
                    except:
                        print("Something's wrong")
                else:
                    try:
                        qua[(i,j,k,j)].append(1)
                    except:
                        print("Something's wrong")
    # Linear terms
    for i in range(N):
        for j in range(N):
            try:
                lin[(i,j,i,j)].append(-4)
            except:
                print("Something's wrong")
    return lin, qua

lin, qua = constraint_hamiltonian(lin,qua)

def group_similar_terms(qua):
    # Group similar terms together eg. X17 * X34 = X34 * X17
    temp = {}
    while bool(qua):
        key = list(qua.keys())
        ref_key = key[0]
        grouped_key = ref_key[2:] + ref_key[:2]

        temp[ref_key] = qua[ref_key] + qua[grouped_key]
        del qua[ref_key]
        del qua[grouped_key]
    
    return temp

qua = group_similar_terms(qua)

def formalize(lin, qua):
    linear = {key[:2] : sum(lin[key]) for key in lin}

    quadratic = {(key[:2], key[2:]) : sum(qua[key]) for key in qua}

    return linear, quadratic

linear, quadratic = formalize(lin,qua)

def scale_bias_couplings(linear, quadratic):
    # IF coupling less than -0.8 : Some problem (check notes)
    
    linear_max = max(np.abs(list(linear.values())))
    quadratic_max = max(np.abs(list(quadratic.values())))

    scaling_factor = max(linear_max, quadratic_max)
    
    if min(list(quadratic.values())) / scaling_factor == -1:
        print("Warning: Check Coupling strengths")

    scaled_linear = {key : linear[key] / scaling_factor for key in linear}
    scaled_quadratic = {key : quadratic[key] / scaling_factor for key in quadratic}
    
    return scaled_linear, scaled_quadratic

scaled_linear, scaled_quadratic = scale_bias_couplings(linear, quadratic)

bqm = dimod.BinaryQuadraticModel(scaled_linear, scaled_quadratic, 0.0, dimod.BINARY)
#sampler = ExactSolver()
#sampler = EmbeddingComposite(DWaveSampler(solver='DW_2000Q_2_1'))
#sampler = VirtualGraphComposite(DWaveSampler(solver='DW_2000Q_2_1'),  chain_strength=2)
#sampler = neal.SimulatedAnnealingSampler()
#response = sampler.sample(bqm, num_reads=5000)

import dwave_qbsolv

reads = 50
sol_limit = 64
system = EmbeddingComposite(DWaveSampler(solver='DW_2000Q_2_1'))
sampler = dwave_qbsolv.QBSolv()
Tref = time()
response = sampler.sample(bqm, num_reads=reads, solver=system, solver_limit=sol_limit, verbosity=1)
Tfin = time()

# Sampling Time

#response = sampler.sample(bqm)
#response = sampler.sample(bqm, num_reads=reads)
#Tfin = time()
#print(Tfin-Tref,'\n')

#for sample,energy,num in response.data(['sample', 'energy', 'num_occurrences']):
#    print(sample, energy, num)


# SOMETHING FISHY HERE (DOUBLE CHECK BEFORE USING)
#sample_matrix = response.samples_matrix
#data_vectors = response.data_vectors

# BEST SOLUTION AVAILABLE RN
best_solution = response.first
solution_data = list(dict(best_solution[0]).values())

# Matrix form of solution
X = np.matrix([solution_data[N*i : N*i + N] for i in range(N)])

def contraint_satisfied(X):
    check = True
    for i in range(N):
        if sum(X[:,i]) != 1 or sum(X[i,:].T) != 1:
            check = False
            break
    return check

Zmin = A * X * B * X.T

print("\nObjective = ", Zmin.trace())
print("Constraints satisifed : ", contraint_satisfied(X))
print("# Reads = ", reads)
print("Solver limit = ", sol_limit)
print("Time taken : ", Tfin-Tref, " sec\n")
print(X,'\n')



# Answer from QAPLIb
#ans = (12,7,9,3,4,8,11,1,5,6,10,2)
#Y = np.matrix([np.zeros(N) for i in range(N)])
#for i in range(len(ans)):
#    Y[i,ans[i]-1] = 1



