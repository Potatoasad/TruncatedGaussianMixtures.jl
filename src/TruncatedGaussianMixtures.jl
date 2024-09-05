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
using DiffRules
using ForwardDiff
using Roots

include("block_structure.jl")
include("TruncatedMvNormal/loglikelihood.jl")
include("TruncatedMvNormal/moments.jl")
include("TruncatedMvNormal/moment_matching.jl")
include("datapreprocess.jl")
include("initialize.jl")
include("ExpectationMaximization.jl")
include("annealing_schedule.jl")
include("Transformations/AbstractTransformation.jl")
include("fit.jl")
include("kde.jl")


export update!, ExpectationMaximization, TruncatedMvNormal, fit_gmm, initialize, AnnealingSchedule
export iterate_to_conclusion, fit_best_truncated_normal_single
export AbstractTransformation, Transformation, forward, inverse, domain_columns, image_columns, sample
export BoundaryUnbiasing, BoundaryUnbiasedData
export fit_kde

end # module TruncatedGaussianMixtures
