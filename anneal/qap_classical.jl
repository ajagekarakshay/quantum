using JuMP;
#using CPLEX;
using Gurobi
#using DataFrames

#import Gurobi

function parse_data(data)
    N = parse(Int8, data[1])
    A = zeros(Int8,N,N);
    B = zeros(Int8,N,N);
    for i=3:N+2
        aline = parse.(Int8, split(data[i]))
        bline = parse.(Int8, split(data[i+N+1]))
        for j=1:N
            A[i-2,j] = aline[j]
            B[i-2,j] = bline[j]
        end
    end
    return N, A, B;
end

function main()
    # Input data
   # file = "nug12";
    file = ARGS[1];
    fp = open(file);
    data = readlines(fp);
    close(fp);
    (N,A,B) = parse_data(data);

    # Define model
    #model = Model(solver=CplexSolver());
    model = Model(solver=GurobiSolver());

    # Vars
    @variable(model, 0 <= x[1:N, 1:N] <= 1, Int);

    # Constraints
    for i=1:N
    @constraint(model, sum{x[i,j], j in 1:N} == 1);
    @constraint(model, sum{x[j,i], j in 1:N} == 1);
    end

    # Objective
    @objective(model, Min, sum{A[i,j] * B[k,l] * x[i,k] * x[j,l], i in 1:N, j in 1:N, k in 1:N, l in 1:N});

    # solve
    @time solution = solve(model);
    println("\n Solution : ",solution,"\n");
    println("Objective : ", getobjectivevalue(model));

end

main()