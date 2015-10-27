include("LocalConstant.jl")
include("LocalPolynomial.jl")
y = rand(100)
X = rand(100,2)
x = rand(1,2)
@time ym, y50, y05, y95 = LocalConstant(y, X, x, 1., true, true)
println("mean, median, q05, q95: ", ym, y50, y05, y95);

@time ym, y50, y05, y95 = LocalPolynomial(y, X, x, 1., true, true)
println("mean, median, q05, q95: ", ym, y50, y05, y95);
