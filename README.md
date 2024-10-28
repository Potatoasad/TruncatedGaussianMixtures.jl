# TruncatedGaussianMixtures.jl
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://potatoasad.github.io/TruncatedGaussianMixtures.jl/index.html)

Allows one to fit a gaussian mixture model using Truncated Gaussian Kernels. Works only for Gaussians truncated to lie inside some box. 

> The algorithm is adapted from [this paper by Lee & Scott](https://www.sciencedirect.com/science/article/abs/pii/S0167947312001156), as well as the algorithm for computing the first two moments of a truncated gaussian with full covariances.

## Advantages

As we can see the standard Gaussian Mixture Model has its kernels avoid the edges. A truncated kernel reproduces the probability distributions at the edges as well.

<img src="./imgs/Comparison2.gif"> </img>

<img src="./imgs/Comparison3.gif"> </img>

## Usage

Let's generate some random data from a known 2D truncated gaussian mixture model:

```julia
using TruncatedGaussianMixtures

# Lets generate some variables
a = [0.0, 0.0]; b = [1.0, 1.0] # Lower and upper limits
μ = [0.5, 0.0]; Σ = Diagonal([0.2, 0.8])
dist = TruncatedMvNormal(MvNormal(μ, Σ), a, b)
X = rand(dist, 8000)
```

Now to fit the data we use the `fit_gmm` function. Below is one method signature we will use with all the keyword arguments. We are going to ask it to fit 2 truncated gaussian components:

```julia
# Create the fit
EM = fit_gmm(X, 2, a, b;   # data, n_components, lower, upper
  cov=:diag,  # Choose between :diag and :full for diagonal or full covariances
  tol=1e-2,   # tolerance for the stopping criteria.
  MAX_REPS=100, # Maximum number of EM update steps
  verbose=false,  # Verbose output usefull for debugging 
  progress=true,  # Gives a progress bar to show the progress of the fit
  responsibilities=false, # Returns the EM object as opposed to Distributions.jl object
  block_structure=false) # One can specify a block structure for the covariances
```

This returns a Distributions.jl Mixture Model object (`Mixture`) with parameters close to those that produced it (i.e. `dist`). 

We can then use `fit_gmm` , and provide it the data samples, number of components and the bounds of the truncated space, along with a covariance block structure we would like. 

```julia
fit = fit_gmm(X, # Provided Data as a 2xN Matrix
    2; # Number of components to fit to
    [0.0,0.0,0.0,0.0], # lower bounds of the truncation
    [1.0,1.0,10.0,11.0]; # upper bounds of the truncation
  	cov=:diag,  # Constrains the truncated mixtures to be diagonal
  	block_structure=[0,0,1,1] #TGMM kernels may be correlated between dimension 3&4 and dimension 1&2, but dimension 1 and 2 may not correlate with dimension 3 and 4
 )
```

## Annealing Schedules

One can use annealing schedules to guide fits better, especially for flatter distributions - though the standard vanilla algorithm does a decent job as well. [Naim & Gildea](https://arxiv.org/abs/1206.6427) propose a deterministic anti-annealing mechanism that alows one to fit better when many of the gaussian components overlap (in the context of non-truncated GMM).

An annealing schedule is a list $\beta_n$​​, such that it changes the responsibilities of the points for a given proposed GMM: 

```math
w^{(n)}_{i k}=\frac{\left(\alpha_k^{(n)} P\left(x_i \mid z_i=k, \Theta^{(n)}\right)\right)^{\beta_n}}{\sum_{m=1}^{K} \left(\alpha_m^{(n)} P\left(x_i \mid z_i=m, \Theta^{(n)}\right)\right)^{\beta_n}}
```

Here is $n$ indexes the EM step made. The standard EM-algorithm is the case where $\beta_n = 1$

### Usage

In `TruncatedGaussianMixtures.jl` the way to add annealing schedules is by instantiating an annealing schedule object with the $\beta_n$ list we would like. For example:

```julia
β = vcat((0.0:0.01:1.5), 1.5:(-0.01):1.0, ones(100))
schedule = AnnealingSchedule(β)
```

We can then use the schedule for fitting the TGMM

```julia
fit_gmm(X, K, a, b, schedule; cov=:full, tol=1e-3, convergence=false)
```

The `convergence=false` parameter prevents the fitting algorithm from stopping based on some convergence criteria and allows the whole schedule to run its course. 
