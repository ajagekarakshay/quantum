import numpy as np
import dwave
from pprint import pprint
import dimod
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite, VirtualGraphComposite
from dwave.system.samplers import DWaveSampler
from time import time
import sys
import dwave_qbsolv

file = "data\\assets50.dat"

fp = open(file,"r")
data = fp.readlines()
fp.close()
data = [x.rstrip() for x in data]

price = []
index = data.index("param Price :=") + 1
while data[index] != ";":
    price.append(float(data[index].split()[1]))
    index += 1

N = len(price)
returns = np.zeros((N, 3))
index = data.index("param: ExpReturn TotalReturn standDev :=") + 1
count = 0
while data[index] != ";":
    line = data[index].split()[1:]
    returns[count,0] = float(line[0]); returns[count,1] = float(line[1]);  returns[count,2] = float(line[2]); 
    index += 1
    count += 1

covar = np.zeros((N,N))
index = data.index("param Covariance default 0 :=") + 1
count = 0
while data[index] != ";":
    for i in range(count,N):
        covar[count, i] = data[index].split()[2]
        index += 1
    count += 1

corel = np.zeros((N,N))
index = data.index("param Correlation default 0 :=") + 1
count = 0
while data[index] != ";":
    for i in range(count,N):
        corel[count, i] = data[index].split()[2]
        index += 1
    count += 1

sampledcovar = np.zeros((N,N))
index = data.index("param SampledCovariance default 0 :=") + 1
count = 0
while data[index] != ";":
    for i in range(count,N):
        sampledcovar[count, i] = data[index].split()[2]
        index += 1
    count += 1


# Converting upper triangular to Symmetric matrices
covar = covar + covar.T - np.multiply(np.eye(N), covar)
sampledcovar = sampledcovar + sampledcovar.T - np.multiply(np.eye(N), sampledcovar)
corel = corel + corel.T - np.multiply(np.eye(N), corel)

# Parameters

budget = 1000000
kappa = 100
returnlevel = (1+0.07)**(1/365) - 1
moneymarket = (1+0.02)**(1/365) - 1
theta = 0.85

ub = 5 # known from presolved problem
R = len(price) # NUmber of risky assets

# Chang value of n to increase or decrease precision
n = 0
ub = ub + n

def prob_hamiltonian():
    # MultiPLIER ADDED AS 0.0001
    A = 0.0001
    lin = {}
    qua = {}
    # Objective function
    for i in range(R):
        for j in range(R):
            for k in range(ub):
                for l in range(ub):
                    mul = covar[i,j] * 2**(l+k-2*n) * (kappa/budget)**2 * price[i] * price[j] * A
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
    weight = 1
    for i in range(R):
        for j in range(R):
            for k in range(ub):
                for l in range(ub):
                    mul = price[i] * price[j] * 2**(k+l-2*n) *(kappa/budget)**2 * weight
                    if i==j and k==l: # Diferrence between quadratic and linear
                        lin[(i,k,j,l)].append(mul)
                    else:
                        qua[(i,k,j,l)].append(mul)

    for i in range(R):
        for k in range(ub):
            mul = -2 * (kappa/budget) * price[i] * 2**(k-n) * weight
            lin[(i,k,i,k)].append(mul)
    
    # Target constraint 1
#    for i in range(R):
#        for j in range(R):
#            for k in range(ub):
#                for l in range(ub):
#                    mul = price[i] * price[j] * (kappa/budget)**2 * 2**(k+l-2*n) * (theta**2 * sampledcovar[i,j] - returns[i,1]*returns[j,1])
#                    if i==j and k==l: # Diferrence between quadratic and linear
#                        lin[(i,k,j,l)].append(mul)
#                    else:
#                        qua[(i,k,j,l)].append(mul)
#    for i in range(R):
#        for k in range(ub):
#            mul = 2 * returnlevel * returns[i,1] * price[i] * (kappa/budget) * 2**(k-n)
#            lin[(i,k,i,k)].append(mul)
#    

#    return lin, qua


    # Derived constraint
    for i in range(R):

        # 1
        for k in range(ub):
            for l in range(ub):
                mul = (theta**2 * sampledcovar[i,i] - returns[i,1])**2 * 2**(k+l-2*n)
                if k==l:
                    lin[(i,k,i,l)].append(mul)
                else:
                    qua[(i,k,i,l)].append(mul)
        
        # 2
        for k in range(ub):
            mul = 4 * returnlevel * (theta**2 * sampledcovar[i,i] - returns[i,1]) * returns[i,1] * 2**(k-n)
            lin[(i,k,i,k)].append(mul)

        # 3
        for j in range(N):
            for k in range(ub):
                for l in range(ub):
                    mul = 2 * (theta**2 * sampledcovar[i,i] - returns[i,1]) * (theta**2 * sampledcovar[i,j] - returns[i,1]*returns[j,1]) * 2**(k+l-2*n)
                    if i==j and k==l:
                        lin[(i,k,j,l)].append(mul)
                    else:
                        qua[(i,k,j,l)].append(mul)

        # 4
        for j in range(N):
            for k in range(ub):
                mul = 4 * returnlevel * returns[i,1] * (theta**2 * sampledcovar[i,j] - returns[i,1]*returns[j,1]) * 2**(k-n)
                lin[(j,k,j,k)].append(mul)

        # 5
        for j in range(N):
            for k in range(N):
                for l in range(ub):
                    for m in range(ub):
                        mul = (theta**2 * sampledcovar[i,j] - returns[i,1]*returns[j,1]) * (theta**2 * sampledcovar[i,k] - returns[i,1]*returns[k,1]) * 2**(l+m-2*n)
                        if j==k and l==m:
                            lin[(j,l,k,m)].append(mul)
                        else:
                            qua[(j,l,k,m)].append(mul)

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
sol_limit = 100
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
Y = np.matrix([solution_data[ub*i : ub*i + ub] for i in range(R)])

def get_solution(Y):
    Z = []
    W = []
    for i in range(R):
        sum = 0
        for k in range(ub):
            sum = sum + Y[i,k] * 2**(k-n)
        Z.append(sum)
        W.append(price[i] * kappa/budget * Z[i])
        
    return (np.array(Z), np.array(W))

# Objective value
def objective_value(W):
    sum = 0
    for i in range(R):
        for j in range(R):
            sum = sum + W[i] * covar[i,j] * W[j] * 1e5
    return sum


def target_constraint_satisfied(W):
    #target
    t_temp1, t_temp2 = 0, 0
    for i in range(R):
        for j in range(R):
            t_temp1 += W[i] * sampledcovar[i,j] * W[j]
    
    for i in range(R):
        t_temp2 += returns[i,1] * W[i]

    return t_temp2 - theta*np.sqrt(t_temp1)



(Z, W) = get_solution(Y)

print("\nTime : ", Tfin-Tref)
print("Variables = Linear terms =", len(lin))
print("Quadratic terms = ", len(qua))
print("Objective : ", objective_value(W))
print("Fraction of budget used = ", sum(W))
print("Target : ", returnlevel, " Achieved target : ", target_constraint_satisfied(W), "  : ", target_constraint_satisfied(Y) >= returnlevel)
print("y_int = ", Z)
print("W = ", W)