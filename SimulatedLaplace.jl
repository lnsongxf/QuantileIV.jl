include("AIS_sl.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")

blas_set_num_threads(1)


function QIVWrapper()
    # features of model
    n = 50  # sample size
    beta = [1., 1.] # true parameters
    tau = 0.5
    # features of estimation procedure
    nParticles = 500 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept
    StopCriterion = 5 # stop when proportion of new particles accepted is below this
    AISdraws = 10000
    mix = 0.5 # proportion drawn from AIS, rest is from prior
    bandwidth = [0.3,0.5] # optimal for RMSE, tuned from prior
    #bandwidth = [0.12, 0.20] # optimal for CI, tuned from prior
    #bandwidth = [0.24, 0.28] # optimal for RMSE, tuned locally
    #bandwidth = [0.135, 0.205] # optimal for CI, tuned locally
    # now implement it
    y,x,z,cholsig,betahatIV = makeQIVdata(beta, tau, n) # draw the data
    otherargs = (y,x,z,tau,cholsig)
    Zn = [0. 0. 0.]
    contrib = AIS_fit(Zn, nParticles, multiples, StopCriterion, AISdraws, mix, otherargs, bandwidth)
end

# the monitoring function
function QIVMonitor(sofar, results)
    if mod(sofar,100) == 0
        theta = [1. 1.]
        # local constant
        m = mean(results[1:sofar,[1;5]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[1;5]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt(mse)
        println()
        println("local constant results")
        println("reps so far: ", sofar)
        println("mean: ", m)
        println("bias: ", b)
        println("st. dev.: ", s)
        println("mse.: ",mse)
        println("rmse.: ",rmse)
        # local linear
        m = mean(results[1:sofar,[9;13]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[9;13]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt(mse)
        println()
        println("local linear results")
        println("reps so far: ", sofar)
        println("mean: ", m)
        println("bias: ", b)
        println("st. dev.: ", s)
        println("mse.: ",mse)
        println("rmse.: ",rmse)
        # simulated Laplace  
        m = mean(results[1:sofar,[17;18]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[17;18]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt(mse)
        println()
        println("Simulated Laplace results")
        println("reps so far: ", sofar)
        println("mean: ", m)
        println("bias: ", b)
        println("st. dev.: ", s)
        println("mse.: ",mse)
        println("rmse.: ",rmse)
         
        inci = (results[1:sofar,3] .< 1.) & (results[1:sofar,4] .>= 1.)
        println()
        println("beta1 CI coverage: ", mean(inci))

        inci = (results[1:sofar,7] .< 1.) & (results[1:sofar,8] .>= 1.)
        println()
        println("beta2 CI coverage: ", mean(inci))
    end

    # save the LL results for local bandwidth tuning
    if sofar == size(results,1)
#        writedlm("first_round_estimates.out", results[:,[9;13]])
    end
end

# dgp
function makeQIVdata(beta, tau, n)
    alpha = [1.0,1.0,1.0]
    varscale = 5.
    alpha = alpha / varscale
    Xi = randn(n,4)
    x = [ones(n,1) Xi[:,1] + Xi[:,2]]
    z = [ones(n,1) Xi[:,2] + Xi[:,3] Xi[:,1] + Xi[:,4]]
    v = randn(n,1)
    epsilon = exp(((z*alpha) .^ 2.).*v) - 1.
    y = x*beta + epsilon
    cholsig = chol(tau*(1. -tau)*(z'*z/n))
    xhat = z*(z\x)
    yhat = z*(z\y)
    betahatIV = inv(x'*xhat)*x'*yhat
    return y,x,z,cholsig,betahatIV
end

# the moments
function aux_stat(beta, otherargs)
    y = otherargs[1]
    x = otherargs[2]
    z = otherargs[3]
    tau = otherargs[4]
    cholsig = otherargs[5]
    m = mean(z.*(tau - map(Float64,y .<= x*beta')),1)
    n = size(y,1);
    m = m + randn(size(m))*cholsig/sqrt(n)
    return m
end


# this function generates a draw from the prior
function sample_from_prior()
	theta = rand(1,2)
    lb = [-1. -1.]
    ub = [3. 3.]
    theta = (ub-lb).*theta + lb
end

# the prior: needed to compute AIS density, which uses
# the prior as a mixture component, to maintain the support
function prior(theta::Array{Float64,2})
    lb = [-1. -1.]
    ub = [3. 3.]
    c = 1./prod(ub - lb)
    p = ones(size(theta,1),1)*c
    ok = all((theta.>=lb) & (theta .<=ub),2)
    p = p.*ok
end    

function check_in_support(theta::Array{Float64,2})
    lb = [-1. -1.]
    ub = [3. 3.]
    ok = all((theta .>= lb) & (theta .<= ub))
    return ok, lb, ub
end


function main()
    MPI.Init()
    reps = 1000   # desired number of MC reps
    n_returns = 18
    pooled = 10  # do this many reps before reporting
    montecarlo(QIVWrapper, QIVMonitor, MPI.COMM_WORLD, reps, n_returns, pooled)
    MPI.Finalize()
end


main()
