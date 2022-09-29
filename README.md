# TruncatedGaussianMixtures.jl
 Allows one to fit a gaussian mixture model using Truncated Gaussian Kernels. 
 Currently only works for Kernels that are constrained to be 
 - uncorrelated
 - 2 dimensional

full correlations and general dimensions will be done in the future as I get time.

## Usage

Let's generate some random data from a known 2D truncated gaussian mixture model:

```julia
# Quick Function to create truncated normal products
TNormal(μ1,σ1,μ2,σ2) = product_distribution(
                                            truncated(Normal(μ1,σ1),0,1),
                                            truncated(Normal(μ2,σ2),0,1)
									)
# Define weights and distributions in mixtures
weights = [0.3,0.7]
dists = [TNormal(0.1,0.4,0.8,0.5),TNormal(0.9,0.1,0.3,0.2)]
X_dist = MixtureModel(dists,weights)

# Generate 8000 samples from this distribution
X = rand(X_dist,8000)

#=
X_dist is:

=#
```

We can then use `fit_gmm_2D` , and provide it the data samples, number of components and the bounds of the truncated space. 

```julia
fit = fit_gmm_2D(X, # Provided Data as a 2xN Matrix
  							 2; # Number of components to fit to
  							bounds1=[0.0,1.0], # x bounds of the truncation
  							bounds2=[0.0,1.0]) # y bounds of the truncation
```

This returns a Distributions.jl Mixture Model with parameters close to those that produced it (i.e. `X_dist`). `X_dist` is:

```julia
#=
(prior = 0.3000): 
(μ=0.1, σ=0.4); lower=0.0, upper=1.0)
(μ=0.8, σ=0.5); lower=0.0, upper=1.0)

(prior = 0.7000): 
(μ=0.9, σ=0.1); lower=0.0, upper=1.0)
(μ=0.3, σ=0.2); lower=0.0, upper=1.0)
=#
```

`fit` is:

```julia
#=
(prior = 0.2891): 
(μ=0.2756150943564681, σ=0.21148968263864146); lower=0.0, upper=1.0)
(μ=0.6176228214141383, σ=0.26340762084394287); lower=0.0, upper=1.0)

(prior = 0.7109): 
(μ=0.8833948163051973, σ=0.0767981133208865); lower=0.0, upper=1.0)
(μ=0.32396173230689124, σ=0.17881611583805454); lower=0.0, upper=1.0)
=#
```

