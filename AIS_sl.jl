#= AIS.jl
These are the generic functions for ABC using adaptive
importance sampling. To use this, you must also supply
the model-specific code, which must define the functions

Z = aux_stat(theta, otherargs): computes aux. stats, given parameter
d = prior(theta): computes prior density at parameter value
theta = sample_from_prior(): generate a draw from prior
ok = check_in_supporttheta): bool checking if param in prior's support

see Auction.jl for an example
=#

using Distributions
using Distances
include("LocalConstant.jl")
include("LocalPolynomial.jl")

# sample from particles: normal perturbation of a 
# single parameter, with bounds enforcement by rejection
# sampling
function sample_from_particles(particles::Array{Float64,2}, delta::Array{Float64,2} )
	n, k = size(particles)
    i = rand(1:n)
	j = rand(1:k)
    ok = false
    theta_s = similar(particles[i,:])
    @inbounds while ok != true
        theta_s = particles[i,:]
        theta_s[:,j] = theta_s[:,j] + delta[:,j]*randn()
        ok, junk, junk = check_in_support(theta_s)
    end    
    return theta_s
end

# sample from AIS density: choose random particle,
# add a MVN perturbation
function sample_from_AIS(particles::Array{Float64,2})
    delta = 1.0*std(particles,1)
	i = rand(1:size(particles,1))
    theta_s = particles[i,:]
    theta_s = theta_s + delta.*randn(size(delta))
    return theta_s
end

# the importance sampling density: mixture of normals
function AIS_density(theta::Array{Float64,2}, particles::Array{Float64,2})
    # what scaling to use here?
    delta  = 1.0*std(particles,1)
    sig = diagm(vec(delta.^2))
    nthetas = size(theta,1)
    nparticles = size(particles,1)
    dens = zeros(nthetas,1)
    @inbounds for i = 1:nparticles
        @inbounds for j = 1:nthetas
            thetaj = theta[j,:]
            thetaj2 = vec(thetaj)
            mu = vec(particles[i,:])
            d = MvNormal(mu, sig)
            dens[j,1] += pdf(d, thetaj2)
        end
    end
    dens = dens/nparticles
    return dens
end    

# the fitting routine: KNN regression, either local linear
# (default) or local constant, weighted by importance sampling
# weights from the AIS density
function AIS_fit(Zn, nParticles, multiples, StopCriterion, AISdraws, mix, otherargs, bandwidth)
    # do AIS to get particles
    particles, Zs = AIS_algorithm(nParticles, multiples, StopCriterion, Zn, otherargs)
    # sample from AIS particles
    thetas = zeros(AISdraws, size(particles,2))
    Zs = zeros(AISdraws, size(Zn,2))
    @inbounds for i = 1:AISdraws
        if rand() < mix
            thetas[i,:] = sample_from_AIS(particles)
        else    
            thetas[i,:] = sample_from_prior()
        end
        Zs[i,:] = aux_stat(thetas[i,:],otherargs)
    end    
    # compute scaling limiting outliers
    cholsig = otherargs[4]
    Zs = Zs*inv(cholsig)
    stdZ = std(Zs,1)
	Zs = Zs ./ stdZ
    if size(bandwidth) == ()
        weights = prod(pdf(Normal(),(Zs.-Zn)/bandwidth),2) # kernel weights
        weights = weights*ones(1,size(thetas,2)) # expand out
    else
        weights = zeros(size(Zs,1),size(thetas,2))
        for i = 1:size(bandwidth,1)
            weights[:,i] = prod(pdf(Normal(),(Zs.-Zn)/bandwidth[i]),2) # kernel weights
        end
    end    
    #AISweights = prior(thetas) ./ ((1. - mix)*prior(thetas) + mix*AIS_density(thetas, particles))
    AISweights = 1.
    weights = AISweights.*weights # overall weights are AIS times kernel weight
    weights = weights ./sum(weights,1)
    test = (weights .> 0.)
    thetas = thetas[test[:,1],:] # the nearest neighbors
    Zs = Zs[test[:,1],:]
    weights = weights[test[:,1],:]
    params = size(thetas,2)
    lc_fit = -999. *ones(1,4*params)
    ll_fit = -999. *ones(1,4*params)
    if size(weights,1) > AISdraws/100.
        for i = 1:params
            lc_fit[:,(i*4-4+1):i*4] = LocalConstant(vec(thetas[:,i]), Zs, Zn, weights[:,i], true, true)
            ll_fit[:,(i*4-4+1):i*4] = LocalPolynomial(vec(thetas[:,i]), Zs, Zn, weights[:,i], true, true)
        end
    end
    n = 50.
    S = size(Zs,1)
    sl_weights = zeros(S,1)
    for i = 1:S
        sl_weights[i,:] = exp(n * -Zs[i,:]*Zs[i,:]'/2.)
    end
    sl_weights = sl_weights/sum(sl_weights,1)
    sl_fit = sum(thetas.*sl_weights,1)
    return [lc_fit ll_fit sl_fit]
end    


# Draw particles from prior
function GetInitialParticles(nParticles::Int64, otherargs)
    particle = sample_from_prior()
    Z = aux_stat(particle, otherargs)
    Zs = zeros(nParticles, size(Z,2))
    particles = zeros(nParticles, size(particle,2))
    particles[1,:] = particle
    Zs[1,:] = Z
    @inbounds for i = 2:nParticles
        particles[i,:] = sample_from_prior()
        Z = aux_stat(particles[i,:], otherargs)
        Zs[i,:]	= Z
    end
    return particles, Zs
end

# Sample from current particles
function GetNewParticles(particles::Array{Float64,2}, multiple::Int64, otherargs)
    nParticles, dimTheta = size(particles)
    newparticles = zeros(multiple*nParticles,dimTheta)
    # get size of Z
    particle = sample_from_prior()
    Z = aux_stat(particle, otherargs)
    dimZ = size(Z,2)
    newZs = zeros(multiple*nParticles, dimZ)
    delta = 0.9*std(particles,1)
    @inbounds for i = 1:multiple*nParticles
        newparticles[i,:] = sample_from_particles(particles, delta)
        newZs[i,:] = aux_stat(newparticles[i,:], otherargs)
    end
    return newparticles, newZs
end

# Select the best particles from current and new
function Select(nParticles, distances::Array{Float64,1}, particles::Array{Float64,2}, Zs::Array{Float64,2})
    ind = sortperm(distances) # indices of distances
    ind = ind[1:nParticles] # indices of best
    new = sum(ind .> nParticles)/nParticles # percentage of new particles kept
    # keep the best distances, particles, Zs
    distances = distances[ind]
    particles = particles[ind,:]
    Zs = Zs[ind,:]
    return new, particles, Zs, distances
end    


#= AIS_algorithm: The adaptive method is used to find the
particles that will be used to construct the importance
sampling density. The particles are found using perturbations
of single parameters, and the particles are kept within user
provided bounds. However, the importance sampling density
that uses the particles is a mixture of normals, each
centered on a particle, and the support is R^k. Thus, no
truncation occurs in the importance sampling density. =# 
function AIS_algorithm(nParticles::Int64, multiple::Int64, StopCriterion, Zn::Array{Float64,2}, otherargs)
    # the initial particles
    particles, Zs = GetInitialParticles(nParticles, otherargs)
    # do bounding to compute scale the first time
    # in the loop, selection will remove outliers
    dimZ = size(Zs,2)
    @inbounds for i = 1:dimZ
        q = quantile(Zs[:,i],0.99)
        # top bound
        test =  Zs[:,i] .< q
        Zs[:,i] = Zs[:,i].*test  + q.*(1. - test)
        q = -quantile(-Zs[:,i],0.99)
        # bottom bound
        test =  Zs[:,i] .> q
        Zs[:,i] = Zs[:,i] .* test + q.*(1. - test)
    end
    iter = 0
    # the main loop
    new = 1.
    #@inbounds while mean(new) > StopCriterion # currently, stops when number of accepted new particles is low enough 
    @inbounds while iter < StopCriterion # currently, stops when number of accepted new particles is low enough 
        iter +=1
        # generate new particles
        newparticles, newZs = GetNewParticles(particles, multiple, otherargs)
        # stack them
        Zs = [Zs; newZs]
        particles = [particles; newparticles]
        # find distances
        cholsig = otherargs[4]
        ZZs = Zs*inv(cholsig)
        stdZ = std(ZZs,1)
        distances = vec(pairwise(Euclidean(),(ZZs./stdZ)', (Zn./stdZ)')) # get all distances
        # select from new and old
        new, particles, Zs, distances =  Select(nParticles, distances, particles, Zs)
        #=if DoPlot & (iter <=5)
            GetPlot(iter, particles, axes, ax)
        end
        =#
    end
    return particles, Zs
end
