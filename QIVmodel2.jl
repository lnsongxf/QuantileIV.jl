using Distributions

# the dgp. tau quantile of epsilon is 0,
# so tau quantile of y is x*beta
function makeQIVdata(beta, tau, n)
    # instruments correlated with regressors
    m = -quantile(Normal(),tau)
    epsilon = quantile(Normal(m),rand(n,1)) # tau quantile is 0
    # component of regressors correlated with inst
    x = randn(n,1)
    # inst
    z = x .+ randn(n,2)
    z = [ones(n,1) z]
    # component of regressors correlated with error
    x = [ones(n,1)  x .+ epsilon]
    b = [epsilon+beta[1] beta[2]*ones(n,1)]
    y = sum(x.*b,2)
    #y = y.*(y.>0.0)
    cholsig = chol(tau*(1.0 -tau)*(z'*z/n))
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
    beta = reshape(beta,2,1)
    m = mean(z.*(tau - map(Float64,y .<= x*beta)),1)
    #m = mean(z.*(tau - map(Float64,y .<= max(0.0,x*beta))),1)
    n = size(y,1);
    m = m + randn(size(m))*cholsig/sqrt(n)
    return m
end

# this function generates a draw from the prior
function sample_from_prior()
	theta = rand(1,2)
    lb = [0.0 0.0]
    ub = [3.0 3.0]
    theta = (ub-lb).*theta + lb
end

# the prior: needed to compute AIS density, which uses
# the prior as a mixture component, to maintain the support
function prior(theta::Array{Float64,2})
    lb = [0.0 0.0]
    ub = [3.0 3.0]
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

