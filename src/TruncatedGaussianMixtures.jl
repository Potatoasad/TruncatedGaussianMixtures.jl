module TruncatedGaussianMixtures

using Distributions
using LogExpFunctions
using Clustering
using LinearAlgebra

using Base: eltype, length
using Distributions: _rand!, rand!, _logpdf, sampler, AbstractRNG
using MvNormalCDF:mvnormcdf
import StatsBase
import StatsBase: Weights
using InvertedIndices
using ProgressMeter
using DataFrames
using DiffRules
using ForwardDiff
using Roots
using Interpolations
using HDF5
using SHA

include("block_structure.jl")
include("TruncatedMvNormal/loglikelihood.jl")
include("TruncatedMvNormal/moments.jl")
include("TruncatedMvNormal/moment_matching.jl")
include("datapreprocess.jl")
include("initialize.jl")
include("ExpectationMaximization.jl")
include("annealing_schedule.jl")
include("Transformations/AbstractTransformation.jl")
include("Transformations/ComposeTransformation.jl")
include("Transformations/QuantileTransformation.jl")
include("fit.jl")
include("kde.jl")
include("tgmm_fit_io.jl")


export update!, ExpectationMaximization, TruncatedMvNormal, fit_gmm, initialize, AnnealingSchedule
export iterate_to_conclusion, fit_best_truncated_normal_single
export AbstractTransformation, Transformation, forward, inverse, domain_columns, image_columns, sample
export QuantileTransformer, fit_quantile_transformer, transform, inverse_transform, compose
export add_quantile_suffix, add_quantile_transformation
export BoundaryUnbiasing, BoundaryUnbiasedData
export fit_kde
export TGMMFit, load_tgmm_h5, save_tgmm_h5, tgmmfit_to_dict, tgmmfit_from_dict

end # module TruncatedGaussianMixtures
