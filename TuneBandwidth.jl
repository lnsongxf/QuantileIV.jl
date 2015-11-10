include("QIVmodel.jl") # load auction model code
include("AIS.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")
blas_set_num_threads(1)


function QIVWrapper()
    # features of model
    n = 200  # sample size
    
    #beta = sample_from_prior() # true parameters
    
    temp = readdlm("first_round_estimates.out")
    i = rand(1:size(temp,1))
    beta = temp[i,:] # true parameters
  
    tau = 0.5
    # features of estimation procedure
    nParticles = 500 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept
    StopCriterion = 5 # stop when proportion of new particles accepted is below this
    AISdraws = 10000
    mix = 0.5 # proportion drawn from AIS, rest is from prior
    bandwidth = 0.1 + rand(0:20)*0.01  
    # now implement it
    y,x,z,cholsig = makeQIVdata(beta', tau, n) # draw the data
    otherargs = (y,x,z,tau,cholsig)
    Zn = [0. 0. 0.]
    contrib = AIS_fit(Zn, nParticles, multiples, StopCriterion, AISdraws, mix, otherargs, bandwidth)
    fitted = [contrib[:,9] contrib[:,13]] # local linear cond. means
    baddata = convert(Float64, any(fitted== -999.))
    error1 = abs((beta - fitted)./(beta+0.1))
    inci = [(contrib[:,3] .< beta[:,1]) & (contrib[:,4] .> beta[:,1]) (contrib[:,7] .< beta[:,2]) & (contrib[:,8] .> beta[:,2]) ] # local linear cond. means
    error2 = inci
    contrib = [baddata bandwidth  error1 error2] 
end

# the monitoring function
function QIVMonitor(sofar, results)
    if mod(sofar,200) == 0
        println(sofar, " done...")
    end
    if sofar == size(results,1)
        ind = sortperm(results[:,2])
        results = results[ind,:]
        writedlm("tune_local.out", results)
    end   
end

function main()
    reps = 1000   # desired number of MC reps
    n_returns = 6
    pooled = 25  # do this many reps before reporting
    montecarlo(QIVWrapper, QIVMonitor, reps, n_returns, pooled)
end


main()
