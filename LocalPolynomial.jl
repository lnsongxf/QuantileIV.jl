# local constant or local linear regression and quantile regression
# still need to add local quadratic

# using the interior point method from QuantileRegression package
include("InteriorPoint.jl") 

function LocalPolynomial(y::Array{Float64,1}, X::Array{Float64,2}, x0::Array{Float64,2}, weights::Array{Float64,1}, order::Int64=1, do_median::Bool=false, do_ci::Bool=false)

test = vec(weights .> 0) # drop obsns with zero weights to help the quantile computations
weights = sqrt(weights[test])
y = y[test]
X = X[test,:]

if order == 0
    n = size(y,1)
    X = reshape(weights,n,1)
    y .*= weights
    ymean = sum(weights.*y)
end    
                 
if order == 1
    X = X .- x0
    X = [ones(size(X,1),1) X]
    X .*= weights
    y .*= weights
    ymean = (X\y)[1,:]
end    

if order == 2
    X = X .- x0
    X = [ones(size(X,1),1) X X.^2.0]
    X .*= weights
    y .*= weights
    ymean = (X\y)[1,:]
end  

if do_median
    y50 = qreg_coef(vec(y), X, 0.5,IP())[1,:]
end

if do_ci
    y05 = qreg_coef(vec(y), X, 0.05,IP())[1,:]
    y95 = qreg_coef(vec(y), X, 0.95,IP())[1,:]
end

if ~do_median && ~do_ci
    return ymean
elseif do_median && ~do_ci
    return [ymean y50]
else    
    return [ymean y50 y05 y95]
end    

end


