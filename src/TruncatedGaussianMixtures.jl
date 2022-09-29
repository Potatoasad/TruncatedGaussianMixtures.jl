module TruncatedGaussianMixtures

using Distributions
using LogExpFunctions
using Clustering

include("TruncatedMixture.jl")
include("initialize.jl")
include("ExpectationMaximization.jl")
include("fit.jl")


export update!, ExpectationMaximization, TruncatedMixture, fit_gmm_2D

end # module TruncatedGaussianMixtures
