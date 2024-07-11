module TruncatedGaussianMixtures

using Distributions
using LogExpFunctions
using Clustering
using LinearAlgebra

using Base: eltype, length
using Distributions: _rand!, _logpdf, sampler, AbstractRNG
using MvNormalCDF:mvnormcdf
import StatsBase
import StatsBase: Weights
using InvertedIndices
using ProgressMeter
using DataFrames

include("block_structure.jl")
include("TruncatedMvNormal/loglikelihood.jl")
include("TruncatedMvNormal/moments.jl")
include("initialize.jl")
include("ExpectationMaximization.jl")
include("annealing_schedule.jl")
include("Transformations/AbstractTransformation.jl")
include("fit.jl")


export update!, ExpectationMaximization, TruncatedMvNormal, fit_gmm, initialize, AnnealingSchedule
export AbstractTransformation, Transformation, forward, inverse, domain_columns, image_columns, sample

end # module TruncatedGaussianMixtures
