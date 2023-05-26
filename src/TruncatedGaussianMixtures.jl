module TruncatedGaussianMixtures

using Distributions
using LogExpFunctions
using Clustering
using LinearAlgebra

using Base: eltype, length
using Distributions: _rand!, _logpdf, sampler, AbstractRNG
using MvNormalCDF:mvnormcdf
using InvertedIndices
using ProgressMeter

include("TruncatedMvNormal/loglikelihood.jl")
include("TruncatedMvNormal/moments.jl")
include("initialize.jl")
include("ExpectationMaximization.jl")
include("fit.jl")


export update!, ExpectationMaximization, TruncatedMvNormal, fit_gmm, initialize

end # module TruncatedGaussianMixtures
