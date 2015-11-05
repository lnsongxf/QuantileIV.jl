include("QIVmodel.jl") # load auction model code
include("AIS.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")

blas_set_num_threads(1)


function QIVWrapper()
    # features of model
    n = 200  # sample size
    beta = [1., 1.] # true parameters
    tau = 0.5
    # features of estimation procedure
    nParticles = 500 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept
    StopCriterion = 0.2 # stop when proportion of new particles accepted is below this
    AISdraws = 10000
    mix = 0.5 # proportion drawn from AIS, rest is from prior
    # now implement it
    y,x,z,cholsig,betahatIV = makeQIVdata(beta, n) # draw the data
    otherargs = (y,x,z,tau,cholsig)
    Zn = [0. 0. 0.]
    contrib = AIS_fit(Zn, nParticles, multiples, StopCriterion, AISdraws, "both", mix,otherargs)
end

# the monitoring function
function QIVMonitor(sofar, results)
    if mod(sofar,1) == 0
        theta = [1. 1.]
        m = mean(results[1:sofar,[1;2]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[1;2]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt(mse)
        println()
        println("parameter 2, local linearresults")
        println("reps so far: ", sofar)
        println("mean: ", m)
        println("bias: ", b)
        println("st. dev.: ", s)
        println("mse.: ",mse)
        println("rmse.: ",rmse)

        m = mean(results[1:sofar,[17;18]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[17;18]],1) 
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
        println("CI coverage: ", mean(inci))
    end
end

function main()
    reps = 1000   # desired number of MC reps
    n_returns = 16
    pooled = 50  # do this many reps before reporting
    montecarlo(QIVWrapper, QIVMonitor, reps, n_returns, pooled)
end


main()
