import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TruncatedGaussianMixtures
using Distributions

TNormal(μ1,σ1,μ2,σ2) = Product([truncated(Normal(μ1,σ1),0,1),truncated(Normal(μ2,σ2),0,1)])

# Define weights and distributions in mixtures
weightss = [0.3,0.7]
dists = [TNormal(0.1,0.4,0.8,0.5),TNormal(0.9,0.1,0.3,0.2)]
X_dist = MixtureModel(dists,weightss)

# Generate 8000 samples from this distribution
X = rand(X_dist,8000)

fit = fit_gmm_2D(X, # Provided Data as a 2xN Matrix
    3; # Number of components to fit to
    bounds1=[0.0,1.0], # x bounds of the truncation
    bounds2=[0.0,1.0], tol=1e-6); # y bounds of the truncation