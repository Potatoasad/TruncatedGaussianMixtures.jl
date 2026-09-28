#=
Small array-level helpers used by the Python wrapper, so it never needs to
`seval` Julia code at runtime.
=#

"""
    truncated_mvnormal(mean, covariance, lower, upper; covariance_type=:full, log_normalizer=nothing)

One `TruncatedMvNormal` from arrays (`covariance` d×d; with
`covariance_type=:diag`, or when d = 1, only its diagonal is used). When `log_normalizer` is
given it is used as the truncation mass instead of being recomputed.
"""
function truncated_mvnormal(mean::AbstractVector, covariance::AbstractMatrix, lower::AbstractVector,
                            upper::AbstractVector; covariance_type=:full, log_normalizer=nothing)
    μ = Vector{Float64}(mean)
    Σ = Matrix{Float64}(covariance)
    # d = 1 always uses the diagonal form (MvNormalCDF needs d ≥ 2; it is the same distribution)
    diagonal = Symbol(covariance_type) == :diag || length(μ) == 1
    normal = diagonal ? MvNormal(μ, sqrt.(diag(Σ))) : MvNormal(μ, Matrix(Symmetric((Σ + Σ') / 2)))
    a = Vector{Float64}(lower)
    b = Vector{Float64}(upper)
    log_normalizer === nothing ? TruncatedMvNormal(normal, a, b) : TruncatedMvNormal(normal, a, b, Float64(log_normalizer))
end

"""
    mixture_from_arrays(means, covariances, weights, lower, upper;
                        log_normalizers=nothing, covariance_type=:full)

Build the legacy `MixtureModel` of `TruncatedMvNormal`s from arrays
(`means` d×K, `covariances` d×d×K). When `log_normalizers` is given it is used
as each component's truncation mass instead of being recomputed.
"""
function mixture_from_arrays(means::AbstractMatrix, covariances::AbstractArray, weights::AbstractVector,
                             lower::AbstractVector, upper::AbstractVector;
                             log_normalizers=nothing, covariance_type=:full)
    comps = map(1:length(weights)) do k
        truncated_mvnormal(view(means, :, k), view(covariances, :, :, k), lower, upper;
                           covariance_type=covariance_type,
                           log_normalizer=log_normalizers === nothing ? nothing : log_normalizers[k])
    end
    w = Vector{Float64}(weights)
    MixtureModel(comps, w ./ sum(w))
end

"""
    mixture_from_components(components, weights)

`MixtureModel` of the given `TruncatedMvNormal`s (e.g. a Python list of them),
with a concrete component type whenever the components share one.
"""
function mixture_from_components(components, weights::AbstractVector)
    w = Vector{Float64}(weights)
    MixtureModel(identity.(collect(components)), w ./ sum(w))
end

"""
    boundary_pseudo_moments(X, lower, upper, structure, bandwidth_scale) -> (M1, V)

Per-observation pseudo moments of the legacy boundary-unbiased data
(`X` is d×N). Returns `M1` (N×d) and the diagonal pseudo variances `V` (N×d);
the legacy second moment is `M1 M1ᵀ + Diagonal(V)` for every observation.
"""
function boundary_pseudo_moments(X::AbstractMatrix, lower::AbstractVector, upper::AbstractVector,
                                 structure::AbstractVector, bandwidth_scale::Real)
    data = BoundaryUnbiasedData(mat2vecs(Matrix{Float64}(X)), Vector{Float64}(lower), Vector{Float64}(upper),
                                BoundaryUnbiasing(collect(structure), bandwidth_scale))
    N, d = size(data.M1)
    V = zeros(N, d)
    for i in 1:d, n in 1:N
        V[n, i] = data.M2[n, i, i] - data.M1[n, i]^2
    end
    return data.M1, V
end
