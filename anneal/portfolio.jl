using JuMP;
using Gurobi;

function main()
    returns = [8 9 12 7];
    covar = [4 3 -1 0
            3 6 1 0
            -1 1 10 0
            0 0 0 0];
    funds = 1000;
    target = 10000;
    model = Model(solver=GurobiSolver());
    @variable(model, x[1:4] >= 0);
    
    @constraint(model, sum(x[i] for i=1:4) == funds);
    @constraint(model, sum(x[i]*returns[i] for i=1:4) == target);
    
    @objective(model, Min, sum(x[i] * sum(x[j]*covar[i,j] for j=1:4) for i=1:4));
    
    @time solve(model);
    println("Objective : ", getobjectivevalue(model));
    println("Solution : ", getvalue(x));
end

function main2()
    mu = [8 9 12 7];
    C = [4 3 -1 0
        3 6 1 0
        -1 1 10 0
        0 0 0 0];
    funds = 10;
    target = 100;

    N = Int64(ceil(log(2,funds)));
    M = length(mu);
    # Change precision here
    n = 4;
    N = N+n;

    model = Model(solver=GurobiSolver());
    @variable(model,  y[1:M, 1:N] , Bin);
    @variable(model, x[1:M]);

    @constraint(model, con[i in 1:M], x[i]==sum(2.0^(k-n-1) * y[i,k] for k=1:N));
    @constraint(model, sum(mu[i] * x[i] for i=1:M) == target );
    @constraint(model, sum(x[i] for i=1:M) == funds);
    

    @objective(model, Min, sum(x[i]*sum(C[i,j]*x[j] for j=1:M) for i=1:M));
    print(model);
    @time solve(model);
    println("Objective : ", getobjectivevalue(model));
    println("Solution : ", getvalue(x));
    println("Solution : ", getvalue(y));
end

main2()