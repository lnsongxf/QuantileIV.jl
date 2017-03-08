# gives local constant conditional mean, and optionally,
# the median and 0.05 and 0.95 quantiles

# using the interior point method from QuantileRegression package
include(Pkg.dir()"/QuantileRegression/src/InteriorPoint.jl") 

using Distributions
using Base.LinAlg.BLAS

function LocalConstant(y, X, x, weights, do_median=false, do_ci=false)
test = weights .> 0 # drop obsns with zero weight
                    # to help the quantile computations
weights = sqrt(weights[test])
y = y[vec(test)]
n = size(y,1)
X = reshape(weights,n,1)
y = weights.*y
ymean = sum(weights.*y)

if do_median
    y50 = qreg_coef(vec(y), X, 0.5,IP())[1]
end

if do_ci
    y05 = qreg_coef(vec(y), X, 0.05,IP())[1]
    y95 = qreg_coef(vec(y), X, 0.95,IP())[1]
end

if ~do_median && ~do_ci
    return ymean
elseif do_median && ~do_ci
    return [ymean y50]
else    
    return [ymean y50 y05 y95]
end    
end
