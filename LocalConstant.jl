# gives local constant conditional mean, and optionally,
# the median and 0.05 and 0.95 quantiles

# using the interior point method from QuantileRegression package
include(Pkg.dir()"/QuantileRegression/src/InteriorPoint.jl") 

using Distributions, Base.LinAlg.BLAS

function LocalConstant(y, X, x, h, do_median=false, do_ci=false)
weights = prod(pdf(Normal(),(X.-x)/h),2) # standard normal product kernel
weights = weights/sum(weights)
test = weights .> 0 # drop obsns with zero weight
                    # to help the quantile computations
weights = sqrt(weights[test])
y = y[vec(test)]
n = size(y,1)
X = reshape(weights,n,1)
y = weights.*y
ymean = sum(weights.*y)

if do_median
    y50 = qreg_ip_coef(y, X, 0.5)
end

if do_ci
    y05 = qreg_ip_coef(y, X, 0.05)
    y95 = qreg_ip_coef(y, X, 0.95)
end

if ~do_median && ~do_ci
    return ymean
elseif do_median && ~do_ci
    return ymean, y50
else    
    return ymean, y50, y05, y95
end    
end
