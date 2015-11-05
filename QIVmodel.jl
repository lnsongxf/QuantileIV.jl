# dgp
function makeQIVdata(beta, n)
    alpha = [1.0,1.0,1.0]
    varscale = 5.
    alpha = alpha / varscale
    tau = 0.5
    Xi = randn(n,4)
    x = [ones(n,1) Xi[:,1] + Xi[:,2]]
    z = [ones(n,1) Xi[:,2] + Xi[:,3] Xi[:,1] + Xi[:,4]]
    v = randn(n,1)
    epsilon = exp(((z*alpha) .^ 2.).*v) - 1.
    y = x*beta + epsilon
    cholsig = chol(tau*(1. -tau)*(z'*z/n))
    #xhat = z*(z\x)
    #yhat = z*(z\y)
    #betahatIV = inv(x'*xhat)*x'*yhat
    #return y,x,z,cholsig,betahatIV
    return y,x,z,cholsig
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
    lb = [0. 0.]
    ub = [3. 3.]
    theta = (ub-lb).*theta + lb
end

# the prior: needed to compute AIS density, which uses
# the prior as a mixture component, to maintain the support
function prior(theta::Array{Float64,2})
    lb = [0. 0.]
    ub = [3. 3.]
    c = 1./prod(ub - lb)
    p = ones(size(theta,1),1)*c
    ok = all((theta.>=lb) & (theta .<=ub),2)
    p = p.*ok
end    

function check_in_support(theta::Array{Float64,2})
    lb = [0. 0.]
    ub = [3. 3.]
    ok = all((theta .>= lb) & (theta .<= ub))
    return ok, lb, ub
end

