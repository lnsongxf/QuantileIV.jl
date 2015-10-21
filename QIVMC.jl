include("QIVmodel.jl") # load auction model code
include("AIS.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")

blas_set_num_threads(1)

function QIVWrapper()
    y,x,z,cholWinv = makeQIVdata()
    tau = 0.5
    global otherargs = (y,x,z,tau,cholWinv)
    Zn = [0. 0. 0.];
    nParticles = 500 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept
    StopCriterion = 0.1 # stop when proportion of new particles accepted is below this
    AISdraws = 1000
    neighbors = 1000
    contrib = AIS_fit(Zn, nParticles, multiples, StopCriterion, AISdraws, neighbors, "both")
end

# the monitoring function
function QIVMonitor(sofar, results)
    if mod(sofar,1) == 0
        theta = [1. 1.]
        m = mean(results[1:sofar,1:end],1)
        er = m - [theta theta]; # theta defined at top level, so ok to use
        b = mean(er,1)
        s = std(results[1:sofar,:],1) 
        mse = s.^2 + b.^2
        rmse = sqrt(mse)
        println()
        println("reps so far: ", sofar)
        println("mean: ", m)
        println("bias: ", b)
        println("st. dev.: ", s)
        println("mse.: ",mse)
        println("rmse.: ",rmse)
    end
end

function main()
    reps = 1000   # desired number of MC reps
    n_returns = 4
    pooled = 50  # do this many reps b
    montecarlo(QIVWrapper, QIVMonitor, reps, n_returns, pooled)
end


main()
