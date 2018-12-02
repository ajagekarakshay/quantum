import numpy as np
import dwave
from pprint import pprint
import dimod
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite, VirtualGraphComposite
from dwave.system.samplers import DWaveSampler
from time import time
import sys
import dwave_qbsolv

# define data
C = np.matrix([[4,3,-1,0], [3,6,1,0], [-1,1,10,0], [0,0,0,0]])  #variance-covariance matrix
mu = [8,9,12,7]   # returns
funds = 10
target = 100
M = len(mu) # Securities
N = int(np.ceil(np.log2(funds))) #Upperbound of each investment

# Chang value of n to increase or decrease precision
n = 4
N = N + n

def prob_hamiltonian():
    # MultiPLIER ADDED AS 0.0001
    lin = {}
    qua = {}
    # Objective function
    for i in range(M):
        for j in range(M):
            for k in range(N):
                for l in range(N):
                    mul = C[i,j] * 2**(l+k-2*n) * 0.0001
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

lin, qua = prob_hamiltonian() 

def constraint_hamiltonian(lin, qua):
    # Funds constraint
    weight1 = 1
    for i in range(M):
        for j in range(M):
            for k in range(N):
                for l in range(N):
                    if i==j and k==l: # Diferrence between quadratic and linear
                        lin[(i,k,j,l)].append(weight1 * 2**(k+l-2*n))
                    else:
                       qua[(i,k,j,l)].append(weight1 * 2**(k+l-2*n))
    for i in range(M):
        for k in range(N):
            lin[(i,k,i,k)].append(-weight1 * 2 * funds * 2**(k-n))
    
    # Target constraint
    for i in range(M):
        for j in range(M):
            for k in range(N):
                for l in range(N):
                    if i==j and k==l: # Diferrence between quadratic and linear
                        lin[(i,k,j,l)].append(mu[i] * mu[j] * 2**(k+l-2*n))
                    else:
                        qua[(i,k,j,l)].append(mu[i] * mu[j] * 2**(k+l-2*n))
    for i in range(M):
        for k in range(N):
            lin[(i,k,i,k)].append(-2 * target * mu[i] * 2**(k-n))
    
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

reads = 1000
sol_limit = 64
system = EmbeddingComposite(DWaveSampler(solver='DW_2000Q_2_1'))
sampler = dwave_qbsolv.QBSolv()
Tref = time()
response = sampler.sample(bqm, num_reads=reads, solver='tabu', solver_limit=sol_limit, verbosity=0)
#response = system.sample(bqm, num_reads=reads)
Tfin = time()

# BEST SOLUTION AVAILABLE RN
best_solution = response.first
solution_data = list(dict(best_solution[0]).values())

# Matrix form of solution
Y = np.matrix([solution_data[N*i : N*i + N] for i in range(M)])

# Objective value
def objective_value(Y):
    sum = 0
    for i in range(M):
        for j in range(M):
            for k in range(N):
                for l in range(N):
                    sum = sum + C[i,j] * 2**(l+k-2*n) * Y[i,k] * Y[j,l]
    return sum

def constraint_satisfied(Y):
    # Funds and target
    f_temp,t_temp = 0,0
    for i in range(M):
        for k in range(N):
            f_temp = f_temp + 2**(k-n) * Y[i,k]
            t_temp = t_temp + mu[i] * 2**(k-n) * Y[i,k]
    return f_temp, t_temp

def get_solution(Y):
    X = []
    for i in range(M):
        sum = 0
        for k in range(N):
            sum = sum + Y[i,k] * 2**(k-n)
        X.append(sum)
    return X

print("\nTime : ", Tfin-Tref)
print("Variables = Linear terms =", len(lin))
print("Quadratic terms = ", len(qua))
print("Objective : ", objective_value(Y))
used_funds, target_achieved = constraint_satisfied(Y)
print("Original Funds : ", funds, " Funds used : ", used_funds)
print("Target : ", target, " Achieved target : ", target_achieved)
print("Solution = ", get_solution(Y))

