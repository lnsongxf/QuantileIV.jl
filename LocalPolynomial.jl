# gives local constant conditional mean, and optionally,
# the median and 0.05 and 0.95 quantiles

# using the interior point method from QuantileRegression package
include("InteriorPoint.jl") 

function LocalPolynomial(y, X, x0, weights, do_median=false, do_ci=false, order=1)
test = weights .> 0 # drop obsns with zero weight
                    # to help the quantile computations
weights = sqrt(weights[test])
y = y[vec(test)]
X = X[vec(test),:]
X = X .- x0

X = [ones(size(X,1),1) X]
X = weights.*X
y = weights.*y
ymean = (X\y)[1,:]

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


