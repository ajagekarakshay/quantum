function main()

    d = [0.04      2.00     2.88
        0.08      1.36     1.32
        0.36      0.08     1.04
        0.88      0.10     0.52  
        1.52      1.80     0.12
        3.36      2.28     0.08 ];
    za = 1.96;
    h = 12;
    L = 7;
    bF = 10;
    chi = 250;
    beta = 0.01;
    seta = 0.01;

    f = [100 100 100];
    g = [1.3 1.0 1.4];
    a = [0.24 0.20 0.28];
    mu = [95; 157; 46; 234; 75; 192];
    sig = [30; 50; 25; 80; 25; 80];

    (i, j) = size(d);
    model = Model(solver=KnitroSolver();)
    @variable(model, N[1:j] >=0);
    @variable(model, demand[1:j] >= 0);
    @variable(model, x[1:j] >= 0, Int);
    @variable(model, y[1:i,1:j] >= 0, Int);
    @variable(model, cost);

    
    @constraint(model, con[m in 1:i], sum(y[m,n] for n=1:j)  == 1);
    @constraint(model, con[m in 1:i, n in 1:j], y[m,n] <= x[n]);
    @constraint(model, con[n in 1:j], demand[n] == chi * sum(mu[m]*y[m,n] for m=1:i));
    @NLconstraint(model,  cost == sum(f[n]*x[n] for n=1:j) + beta * sum(chi*d[m,n]*mu[m]*y[m,n] for m=1:i,n=1:j)
                        + sum(x[n] * (bF*N[n] + beta*g[n]*N[n] + beta*a[n]*demand[n] + seta*h*demand[n]/(2*N[n])) for n=1:j)
                       + seta*h*za*sum(sqrt(sum(L*sig[m]^2*y[m,n] for m=1:i)) for n=1:j) );

    @NLobjective(model, Min, cost);
    print(model)

    solution = solve(model)
    print(solution);
end

main()