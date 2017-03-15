using QuantileRegression
include("QIVmodel3.jl") # load auction model code
include("AIS.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")

#blas_set_num_threads(1)


function QIVWrapper()
    # features of model
    n = 200  # sample size
    beta = [1.0, 1.0] # true parameters
    tau = 0.5
    # features of estimation procedure
    nParticles = 200 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept
    StopCriterion = 5 # stop when proportion of new particles accepted is below this
    AISdraws = 10000
    mix = 0.5 # proportion drawn from AIS, rest is from prior
    bandwidth = [0.1, 0.1] # guess with more stats
    #bandwidth = [0.17,0.17] # optimal for RMSE, tuned from prior
    #bandwidth = [0.12, 0.20] # optimal for CI, tuned from prior
    #bandwidth = [0.24, 0.28] # optimal for RMSE, tuned locally
    #bandwidth = [0.135, 0.20] # optimal for CI, tuned locally
    # now implement it
    y,x,z,betahatIV = makeQIVdata(beta, tau, n) # draw the data
    otherargs = (y,x,z,tau)
    m = mean(z.*(tau - map(Float64,y .<= max(0.0, x*beta))),1)
    Zn = [m mean(y.==0.0) betahatIV'] 
    contrib = AIS_fit(Zn, nParticles, multiples, StopCriterion, AISdraws, mix, otherargs, bandwidth)
    contrib = [contrib betahatIV']
end

# the monitoring function
function QIVMonitor(sofar::Int64, results::Array{Float64,2})
    if mod(sofar,10) == 0
        theta = [1.0 1.0]
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
        # IV  
        m = mean(results[1:sofar,[17;18]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[1;2]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt(mse)
        println()
        println("IV results")
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

function main()
    MPI.Init()
    reps = 1000   # desired number of MC reps
    n_returns = 18
    pooled = 10  # do this many reps before reporting
    montecarlo(QIVWrapper, QIVMonitor, MPI.COMM_WORLD, reps, n_returns, pooled)
    MPI.Finalize()
end


main()
