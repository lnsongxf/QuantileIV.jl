using Distributions

# the dgp. tau quantile of epsilon is 0,
# so tau quantile of y is x*beta
function makeQIVdata(beta::Array{Float64,1}, tau::Float64, n::Int64)
    # instruments correlated with regressors
    m = -quantile(Normal(),tau)
    epsilon = quantile(Normal(m),rand(n,1)) # tau quantile is 0
    # inst
    z = [ones(n,1) randn(n,2)]
    # endogenous regressor from reduced form
    alpha = [1.0, 1.0, 1.0] # rf coefficients
    x = z*alpha + 0.5*epsilon + 0.5*randn(n,1) # add rf error, which depends on epsilon
    x = [ones(n,1) x]
    # structural eqn for endog of interest
    b = hcat(epsilon.+beta[1], beta[2]*ones(n,1))
    y = sum(x.*b,2)
    y = y.*(y.>0.0)
    xhat = z*(z\x)
    yhat = z*(z\y)
    betahatIV = inv(x'*xhat)*x'*yhat
    return y,x,z,betahatIV
end

# the moments
function aux_stat(beta::Array{Float64,1}, otherargs)
    y = otherargs[1]
    x = otherargs[2]
    z = otherargs[3]
    tau = otherargs[4]
    n = 200
    m = mean(z.*(tau - map(Float64,y .<= max(0.0,x*beta))),1)
    y,x,z,betahatIV = makeQIVdata(beta, tau, n) # draw the data
    mm = [m mean(y.==0.0) betahatIV'] 
    return mm
end

# this function generates a draw from the prior
function sample_from_prior()
	theta = rand(2)
    lb = [0.0; 0.0]
    ub = [3.0; 3.0]

    theta = (ub-lb).*theta + lb
end

# the prior: needed to compute AIS density, which uses
# the prior as a mixture component, to maintain the support
function prior(theta::Array{Float64,1})
    lb = [0.0; 0.0]
    ub = [3.0; 3.0]
    c = 1./prod(ub - lb)
    p = ones(size(theta,1),1)*c
    ok = all((theta.>=lb) & (theta .<=ub),2)
    p = p.*ok
end    

function check_in_support(theta::Array{Float64,1})
    lb = [0.; 0.]
    ub = [3.; 3.]
    ok = all((theta .>= lb) & (theta .<= ub))
    return ok, lb, ub
end

