include("QIVmodel.jl") # load auction model code
include("AIS.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")
blas_set_num_threads(1)


function QIVWrapper()
    # features of model
    n = 200  # sample size
    beta = sample_from_prior() # true parameters
    tau = 0.5
    # features of estimation procedure
    nParticles = 500 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept
    StopCriterion = 0.2 # stop when proportion of new particles accepted is below this
    AISdraws = 10000
    mix = 0.5 # proportion drawn from AIS, rest is from prior
    bandwidth = 0.04 + rand(0:50)*0.01  # bws are random from 0.04 to 0.54
    # now implement it
    y,x,z,cholsig = makeQIVdata(beta', n) # draw the data
    otherargs = (y,x,z,tau,cholsig)
    Zn = [0. 0. 0.]
    contrib = AIS_fit(Zn, nParticles, multiples, StopCriterion, AISdraws, mix, otherargs, bandwidth)
    baddata = convert(Float64, any(contrib== -999.))
    fitted = [(contrib[:,3] .< beta[:,1]) & (contrib[:,4] .> beta[:,1]) (contrib[:,7] .< beta[:,2]) & (contrib[:,8] .> beta[:,2]) ] # local linear cond. means
    error = abs(fitted .-0.9)
    # return bandwidth and squared error for LL cond. mean, estimator
    contrib = [baddata bandwidth  error] 
end

# the monitoring function
function QIVMonitor(sofar, results)
    if mod(sofar,200) == 0
        println(sofar, " done...")
    end
    if sofar == size(results,1)
        ind = sortperm(results[:,2])
        results = results[ind,:]
        writedlm("tune_from_prior_CI.out", results)
    end   
end

function main()
    reps = 1000   # desired number of MC reps
    n_returns = 4
    pooled = 10  # do this many reps before reporting
    montecarlo(QIVWrapper, QIVMonitor, reps, n_returns, pooled)
end


main()
