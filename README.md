# QuantileIV.jl
Quantile instrumental variables estimation using ABC of GMM

This repository contains code for the quantile instrumental variables example in "Bayesian Indirect Inference
and the ABC of GMM" by Creel, Gao, Hong and Kristensen http://arxiv.org/abs/1512.07385 The MPI.jl, Distances.jl and Distributions.jl packages are used.


For replication:

1. run "mpirun -np 21 julia TuneBandwidthFromPrior.jl" to generate
the squared errors corresponding to different bandwidths, using draws
from the prior
2. run "octave-cli --eval Analyze" to analyze those results, and 
compute the optimal bandwidths
3. do the same thing with the "CI" versions of both programs, to get
bandwidths for confidence intervals.
4. run "mpirun -np 21 julia QIVMC.jl" to get the results, having selected
the bandwidthss by editing the program
