# TruncatedGaussianMixtures.jl
 Allows one to fit a gaussian mixture model using Truncated Gaussian Kernels. Works only for Gaussians truncated to lie inside some box

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

# Create the fit
EM = fit_gmm(X, 2, a, b;   # data, n_kernels, lower, upper
  cov=:diag,  # Choose between :diag and :full for diagonal or full covariances
  tol=1e-2,   # tolerance for the stopping criteria.
  MAX_REPS=100, # Maximum number of EM update steps
  verbose=false,  # Verbose output usefull for debugging 
  progress=true,  # Gives a progress bar to show the progress of the fit
  responsibilities=false, # Returns the EM object as opposed to Distributions.jl object
  block_structure=false) # One can specify a block structure for the covariances
```

We can then use `fit_gmm_2D` , and provide it the data samples, number of components and the bounds of the truncated space. 

```julia
fit = fit_gmm(X, # Provided Data as a 2xN Matrix
    2; # Number of components to fit to
    [0.0,0.0], # lower bounds of the truncation
    [1.0,1.0], # upper bounds of the truncation
  	cov=:diag  # Constrains the truncated mixtures to be diagonal
 )
```

## Annealing Schedules

One can use annealing schedules to guide fits better, especially for flatter distributions. [Naim & Gildea](https://arxiv.org/abs/1206.6427) propose a deterministic anti-annealing mechanism that alows one to fit better when many of the gaussian components overlap (in the context of non-truncated GMM).

An annealing schedule is a list $\beta_n$, such that it changes the responsibilities of the points for a given proposed GMM:
$$
w^{(n)}_{i k}=\frac{\left(\alpha_k^{(t)} P\left(x_i \mid z_i=k, \Theta^{(t)}\right)\right)^{\beta_n}}{\sum_{m=1}^K\left(\alpha_m^{(t)} P\left(x_i \mid z_i=m, \Theta^{(t)}\right)\right)^{\beta_n}}
$$
Here is $n$ is the number of EM steps made. The standard EM-algorithm is the case where $\beta_n = 1$

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
