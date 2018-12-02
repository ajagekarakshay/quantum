using JuMP
using KNITRO
using Gurobi
using Pavito
using LinearAlgebra
using AmplNLWriter

    file = "data\\assets50.dat";

    fp = open(file,"r");
    data = readlines(fp);
    close(fp);

    price = [];
    index = findfirst(isequal("param Price :="), data) + 1;
    while data[index] != ";"
        append!(price, parse(Float64, split(data[index])[2]));
        global index += 1;
    end

    N = length(price);
    returns = zeros(N, 3);
    index = findfirst(isequal("param: ExpReturn TotalReturn standDev :="), data) + 1;
    count = 1;
    while data[index] != ";"
        line = parse.(Float64, split(data[index])[2:end]);
        returns[count,1] = line[1]; returns[count,2] = line[2];  returns[count,3] = line[3]; 
        global index += 1;
        global count += 1;
    end

    
    covar = zeros(N,N);
    index = findfirst(isequal("param Covariance default 0 :="), data) + 1;
    count = 1;
    while data[index] != ";"
        for i=count:N
            covar[count, i] = parse(Float64, split(data[index])[3]);
            global index += 1;
        end
        global count += 1;
    end
    
    corel = zeros(N,N);
    index = findfirst(isequal("param Correlation default 0 :="), data) + 1;
    count = 1;
    while data[index] != ";"
        for i=count:N
            corel[count, i] = parse(Float64, split(data[index])[3]);
            global index += 1;
        end
        global count += 1;
    end

    sampledcovar = zeros(N,N);
    index = findfirst(isequal("param SampledCovariance default 0 :="), data) + 1;
    count = 1;
    while data[index] != ";"
        for i=count:N
            sampledcovar[count, i] = parse(Float64, split(data[index])[3]);
            global index += 1;
        end
        global count += 1;
    end

# COnverting upper triangular to symmetric matrices

covar = covar + transpose(covar) - diagm(0 => diag(covar));
sampledcovar = sampledcovar + transpose(sampledcovar) - diagm(0 => diag(sampledcovar));
corel = corel + transpose(corel) - diagm(0 => diag(corel));


function main()
    budget = 1000000;
    kappa = 100;
    returnlevel = (1+0.07)^(1/365) - 1;
    moneymarket = (1+0.02)^(1/365) - 1;
    theta = 0.85;

   # model = Model(solver=PavitoSolver(cont_solver=KnitroSolver() ,mip_solver=GurobiSolver()));
   model = Model(solver=KnitroSolver())
   # model = Model(solver=GurobiSolver()); 
    @variable(model, w[1:N] >= 0);
    @variable(model, w0 >= 0);
    @variable(model, 31 >=y[1:N] >= 0, Int);

    @constraint(model, cons[j in 1:N], w[j] == price[j]*y[j]*kappa / budget);
    @constraint(model, w0 + sum(w[j] for j=1:N) == 1);
    @constraint(model, w0==0);
   # @NLconstraint(model, moneymarket*w0 + sum(returns[j,1]*w[j] for j=1:N) -theta * sqrt(sum(2*w[i]*sum(w[j] * sampledcovar[i,j] for j=i+1:N) for i=1:N) + sum(w[i]*w[i]*sampledcovar[i,i] for i=1:N)) >= returnlevel);
  #  @NLconstraint(model, theta^2 * sum(w[i]*sum(w[j] * sampledcovar[i,j] for j=1:N) for i=1:N) <= (-returnlevel + moneymarket*w0 + sum(returns[j,1]*w[j] for j=1:N))^2 );
    @constraint(model, cons[i in 1:N], 4 * returnlevel^2 * returns[i,1]^2 + (theta^2 * sampledcovar[i,i] - returns[i,1])^2 * w[i]^2 + 4 * returnlevel * (theta^2 * sampledcovar[i,i] - returns[i,1]) * returns[i,1] * w[i] + 2 * (theta^2 * sampledcovar[i,i] - returns[i,1]) * w[i] * sum((theta^2 * sampledcovar[i,j] - returns[i,1]*returns[j,1]) * w[j] for j=1:N) + 4 * returnlevel * returns[i,1] * sum((theta^2 * sampledcovar[i,j] - returns[i,1]*returns[j,1]) * w[j] for j=1:N) + sum(sum((theta^2 * sampledcovar[i,j] - returns[i,1]*returns[j,1]) * (theta^2 * sampledcovar[i,k] - returns[i,1]*returns[k,1]) * w[j] * w[k] for k=1:N) for j=1:N) == 0);
    @objective(model, Min, 1e5 * (sum(w[i]*sum(w[j] * covar[i,j] for j=1:N) for i=1:N) ));
   # print(model);

    @time solve(model);
    println("Objective : ", getobjectivevalue(model));
    println("w0 = ", getvalue(w0));
    println("w = ", getvalue(w));
    println("y = ", getvalue(y));
    w = getvalue(w);
    target_achieved = sum(returns[i,1]*w[i] for i=1:N) - theta * sqrt(sum(w[i]*sum(w[j] * sampledcovar[i,j] for j=1:N) for i=1:N));
    funds_used = sum(w[j] for j=1:N);
    println("target_achieved = ", target_achieved); 
    println("Fraction of funds used = ", funds_used);
end


#main()
ub = 5
function approx()
    budget = 1000000;
    kappa = 100;
    returnlevel = (1+0.07)^(1/365) - 1;
    moneymarket = (1+0.02)^(1/365) - 1;
    theta = 0.85;
    k = kappa/budget;

    #model = Model(solver=GurobiSolver());
    model = Model(solver=KnitroSolver());
    #model = Model(solver=AmplNLSolver("couenne"));

    @variable(model, y[1:N, 1:ub], Bin);
    @variable(model, Y[1:N]);
   # @variable(model, W[1:N]);

    #@constraint(model, con[i in 1:N], W[i] == price[i] * kappa/budget * Y[i]);
    @constraint(model, con[i in 1:N], Y[i] == sum(2^(k-1) * y[i,k] for k=1:ub));
    @constraint(model, sum(price[i] * Y[i] * k for i=1:N) == 1);
    @NLconstraint(model, (sum(returns[i,1] * price[i] * Y[i] * k for i=1:N) - returnlevel)^2 >= theta^2 * sum(price[i]*Y[i]*k * sum(sampledcovar[i,j]*price[j]*Y[j]*k for j=1:N) for i=1:N));
    

    @objective(model, Min, 1e5 * sum(price[i] * Y[i] * k * sum(covar[i,j] * price[j] * Y[j] * k for j=1:N) for i=1:N));

    @time solve(model);
    W = k * price .* getvalue(Y);
    println("Objective : ", getobjectivevalue(model));
    println("W = ", W);
    println("y = ", getvalue(y));
    println("Y = ", getvalue(Y))
    
    target_achieved = sum(returns[i,1]*W[i] for i=1:N) - theta * sqrt(sum(W[i]*sum(W[j] * sampledcovar[i,j] for j=1:N) for i=1:N));
    funds_used = sum(W[j] for j=1:N);
    println("target_achieved = ", target_achieved); 
    println("Fraction of funds used = ", funds_used);


end

#approx()