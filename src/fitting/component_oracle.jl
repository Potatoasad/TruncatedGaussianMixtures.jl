#=
Batched, threaded, array-level moment oracle.

Layout conventions (Julia side, column major):
    means            d × K
    covariances      d × d × K   (cross-block entries ignored / zero)
    packed blocks    P × K       with P = Σ_b n_b²; block b occupies rows
                                 offset_b+1 : offset_b+n_b², column-major within the block

One call evaluates all K components; components are split into contiguous
chunks, each processed by one task owning its own workspace.
=#

"""
    canonical_blocks(labels) -> Vector{Vector{Int}}

Blocks of equal labels, ordered by first appearance (deterministic, unlike the
Dict-ordered `CovarianceBlockStructure`).
"""
function canonical_blocks(labels::AbstractVector)
    order = Any[]
    groups = Dict{Any,Vector{Int}}()
    for (i, l) in enumerate(labels)
        if !haskey(groups, l)
            push!(order, l)
            groups[l] = Int[]
        end
        push!(groups[l], i)
    end
    return [groups[l] for l in order]
end

struct MomentOraclePlan
    d::Int
    block_indices::Vector{Vector{Int}}
    block_plans::Vector{BlockMomentPlan}
    packed_offsets::Vector{Int}
    packed_length::Int
end

function MomentOraclePlan(block_indices::Vector{Vector{Int}})
    d = sum(length, block_indices)
    sort(reduce(vcat, block_indices)) == collect(1:d) ||
        throw(ArgumentError("blocks must partition 1:$d"))
    plans_by_size = Dict{Int,BlockMomentPlan}()
    plans = [get!(() -> BlockMomentPlan(length(b)), plans_by_size, length(b)) for b in block_indices]
    offsets = cumsum([0; [length(b)^2 for b in block_indices]])
    MomentOraclePlan(d, block_indices, plans, offsets[1:end-1], offsets[end])
end

"""
    MomentOraclePlan(d, block_labels, covariance_type)

`covariance_type = :diag` puts every coordinate in its own block; `:full` uses
`block_labels` (all one block when `block_labels` is `nothing`/`false`).
"""
function MomentOraclePlan(d::Integer, block_labels, covariance_type::Symbol)
    if covariance_type == :diag
        return MomentOraclePlan([[i] for i in 1:d])
    elseif covariance_type == :full
        if block_labels === nothing || block_labels === false
            return MomentOraclePlan([collect(1:d)])
        end
        length(block_labels) == d || throw(ArgumentError("block structure must have length $d"))
        return MomentOraclePlan(canonical_blocks(collect(block_labels)))
    else
        throw(ArgumentError("unknown covariance type $covariance_type (use :full or :diag)"))
    end
end

struct OracleWorkspace{T}
    blocks::Vector{BlockMomentWorkspace{T}}
end

OracleWorkspace{T}(plan::MomentOraclePlan; qmc_points_per_dim::Int=1000) where {T} =
    OracleWorkspace{T}([BlockMomentWorkspace{T}(length(b); qmc_points_per_dim=qmc_points_per_dim) for b in plan.block_indices])

"""
    OracleContext(d, K, block_labels, covariance_type; kwargs...)

Owns the plan, one workspace per task chunk, and the result arrays of the last
oracle evaluation (together with a copy of the parameters they belong to, so
that the next M-step can reuse them).
"""
mutable struct OracleContext{T}
    plan::MomentOraclePlan
    K::Int
    workspaces::Vector{OracleWorkspace{T}}
    threaded::Bool
    specialize::Bool
    cdf_rtol::T
    seed::UInt64
    # parameters of the last evaluation
    cached_means::Matrix{T}
    cached_covariances::Array{T,3}
    cached_lower::Vector{T}
    cached_upper::Vector{T}
    cache_valid::Bool
    # results of the last evaluation
    log_normalizers::Vector{T}
    centered_means::Matrix{T}       # d × K  E[Y],     Y = X - μ
    centered_second::Matrix{T}      # P × K  E[YYᵀ] blocks
    model_means::Matrix{T}          # d × K  E[X]
    model_second::Matrix{T}         # P × K  E[XXᵀ] blocks
    status::Vector{Symbol}
    max_cdf_relative_error::Vector{T}
    # scratch owned by the context (host inputs are copied here)
    in_means::Matrix{T}
    in_covariances::Array{T,3}
    in_lower::Vector{T}
    in_upper::Vector{T}
    n_evaluations::Int
end

function OracleContext(d::Integer, K::Integer, block_labels=nothing, covariance_type=:full;
                       threaded::Bool=true, specialize::Bool=true, cdf_rtol::Real=0.05,
                       qmc_points_per_dim::Integer=1000, seed::Integer=0, T::Type=Float64)
    covariance_type = Symbol(covariance_type)
    plan = MomentOraclePlan(d, block_labels, covariance_type)
    nws = max(1, Threads.nthreads())
    wss = [OracleWorkspace{T}(plan; qmc_points_per_dim=qmc_points_per_dim) for _ in 1:nws]
    P = plan.packed_length
    OracleContext{T}(plan, K, wss, threaded, specialize, T(cdf_rtol), UInt64(seed),
        zeros(T, d, K), zeros(T, d, d, K), zeros(T, d), zeros(T, d), false,
        zeros(T, K), zeros(T, d, K), zeros(T, P, K), zeros(T, d, K), zeros(T, P, K),
        fill(:not_evaluated, K), zeros(T, K),
        zeros(T, d, K), zeros(T, d, d, K), zeros(T, d), zeros(T, d), 0)
end

"""
    run_chunked!(f, ctx)

Call `f(k, workspace)` for every component `k`, splitting components into
contiguous chunks processed by independent tasks (one workspace per chunk).
"""
function run_chunked!(f::F, ctx::OracleContext) where {F}
    K = ctx.K
    nchunks = ctx.threaded ? min(K, length(ctx.workspaces)) : 1
    if nchunks <= 1
        serial_ws = ctx.workspaces[1]
        for k in 1:K
            f(k, serial_ws)
        end
        return nothing
    end
    chunks = Iterators.partition(1:K, cld(K, nchunks))
    tasks = map(enumerate(chunks)) do (c, ks)
        # `local` matters: each task must own its binding to its workspace
        Threads.@spawn begin
            local task_ws = ctx.workspaces[c]
            for k in ks
                f(k, task_ws)
            end
        end
    end
    foreach(wait, tasks)
    return nothing
end

"""
    component_moments!(ctx, k, ws, means, covariances, lower, upper) -> status

Evaluate log Z and the (centered and noncentered) block moments of component
`k`, writing into column `k` of the context result arrays.
"""
function component_moments!(ctx::OracleContext{T}, k::Int, ows::OracleWorkspace{T},
                            means::AbstractMatrix{T}, covariances::AbstractArray{T,3},
                            lower::AbstractVector{T}, upper::AbstractVector{T}) where {T}
    plan = ctx.plan
    logZ = zero(T)
    status = :ok
    maxerr = zero(T)
    @inbounds for (b, idx) in enumerate(plan.block_indices)
        ws = ows.blocks[b]
        bplan = plan.block_plans[b]
        n = bplan.size
        # reseed per (component, block) so results do not depend on thread layout
        # (size-1 blocks never touch the RNG; `seed!` itself allocates)
        n ≥ 2 && seed!(ws.rng, hash((ctx.seed, k, b)))
        ws.max_cdf_relative_error = zero(T)
        for (jj, j) in enumerate(idx)
            μj = means[j, k]
            ws.lo[jj] = lower[j] - μj
            ws.hi[jj] = upper[j] - μj
            for (ii, i) in enumerate(idx)
                ws.Σ[ii, jj] = covariances[i, j, k]
            end
        end
        st = centered_moments!(ws, bplan; specialize=ctx.specialize)
        maxerr = max(maxerr, ws.max_cdf_relative_error)
        if st !== :ok
            status = st
            logZ = T(NaN)
            break
        end
        logZ += ws.cache.logZ
        off = plan.packed_offsets[b]
        for (jj, j) in enumerate(idx)
            e_j = ws.m1[jj]
            μj = means[j, k]
            ctx.centered_means[j, k] = e_j
            ctx.model_means[j, k] = e_j + μj
            for (ii, i) in enumerate(idx)
                e_i = ws.m1[ii]
                μi = means[i, k]
                q = ws.m2[ii, jj]
                p = off + (jj - 1) * n + ii
                ctx.centered_second[p, k] = q
                # E[XXᵀ] = Q + μ eᵀ + e μᵀ + μ μᵀ
                ctx.model_second[p, k] = q + μi * e_j + e_i * μj + μi * μj
            end
        end
    end
    if status === :ok && maxerr > ctx.cdf_rtol
        status = :cdf_error_above_tolerance
    end
    ctx.log_normalizers[k] = logZ
    ctx.max_cdf_relative_error[k] = maxerr
    ctx.status[k] = status
    return status
end

function _copy_inputs!(ctx::OracleContext, means, covariances, lower, upper)
    d, K = size(means)
    d == ctx.plan.d || throw(DimensionMismatch("means must be $(ctx.plan.d) × K, got $(size(means))"))
    size(covariances) == (d, d, K) || throw(DimensionMismatch("covariances must be $d × $d × $K, got $(size(covariances))"))
    (length(lower) == d && length(upper) == d) || throw(DimensionMismatch("bounds must have length $d"))
    if K != ctx.K
        _resize!(ctx, K)
    end
    copyto!(ctx.in_means, means)
    copyto!(ctx.in_covariances, covariances)
    copyto!(ctx.in_lower, lower)
    copyto!(ctx.in_upper, upper)
    return nothing
end

function _resize!(ctx::OracleContext{T}, K::Int) where {T}
    d = ctx.plan.d
    P = ctx.plan.packed_length
    ctx.K = K
    ctx.cached_means = zeros(T, d, K); ctx.cached_covariances = zeros(T, d, d, K)
    ctx.cache_valid = false
    ctx.log_normalizers = zeros(T, K); ctx.centered_means = zeros(T, d, K)
    ctx.centered_second = zeros(T, P, K); ctx.model_means = zeros(T, d, K)
    ctx.model_second = zeros(T, P, K); ctx.status = fill(:not_evaluated, K)
    ctx.max_cdf_relative_error = zeros(T, K)
    ctx.in_means = zeros(T, d, K); ctx.in_covariances = zeros(T, d, d, K)
    return ctx
end

_cache_matches(ctx::OracleContext) =
    ctx.cache_valid && ctx.cached_means == ctx.in_means && ctx.cached_covariances == ctx.in_covariances &&
    ctx.cached_lower == ctx.in_lower && ctx.cached_upper == ctx.in_upper

# evaluate the oracle at ctx.in_* (reusing the previous evaluation if identical)
function _evaluate_inputs!(ctx::OracleContext)
    _cache_matches(ctx) && return ctx
    means, covs, lo, hi = ctx.in_means, ctx.in_covariances, ctx.in_lower, ctx.in_upper
    run_chunked!((k, ws) -> component_moments!(ctx, k, ws, means, covs, lo, hi), ctx)
    copyto!(ctx.cached_means, means)
    copyto!(ctx.cached_covariances, covs)
    copyto!(ctx.cached_lower, lo)
    copyto!(ctx.cached_upper, hi)
    ctx.cache_valid = true
    ctx.n_evaluations += 1
    return ctx
end

"""
    component_oracle!(ctx, means, covariances, lower, upper)

Evaluate all components in one call. Returns a NamedTuple of the context's
result arrays (log normalizers, model means, packed model second moments,
statuses, ...). The arrays are owned by the context and overwritten by the next call.
"""
function component_oracle!(ctx::OracleContext, means::AbstractMatrix, covariances::AbstractArray,
                           lower::AbstractVector, upper::AbstractVector)
    _copy_inputs!(ctx, means, covariances, lower, upper)
    _evaluate_inputs!(ctx)
    return oracle_result(ctx)
end

oracle_result(ctx::OracleContext) = (
    log_normalizers=ctx.log_normalizers,
    model_means=ctx.model_means,
    model_second_moments=ctx.model_second,
    centered_means=ctx.centered_means,
    centered_second_moments=ctx.centered_second,
    status=String.(ctx.status),
    success=all(==(:ok), ctx.status),
    max_cdf_relative_error=ctx.max_cdf_relative_error,
)

"""
    component_oracle(means, covariances, lower, upper, block_labels=nothing; covariance_type=:full, kwargs...)

Allocating convenience wrapper around [`component_oracle!`](@ref).
"""
function component_oracle(means::AbstractMatrix, covariances::AbstractArray, lower::AbstractVector,
                          upper::AbstractVector, block_labels=nothing; covariance_type=:full, kwargs...)
    d, K = size(means)
    ctx = OracleContext(d, K, block_labels, covariance_type; kwargs...)
    r = component_oracle!(ctx, means, covariances, lower, upper)
    return map(x -> x isa AbstractArray ? copy(x) : x, r)
end

"""
    unpack_blocks(packed, plan) -> Array{T,3}

Dense d × d × K representation of packed block matrices (cross-block entries zero).
"""
function unpack_blocks(packed::AbstractMatrix{T}, plan::MomentOraclePlan) where {T}
    d = plan.d
    K = size(packed, 2)
    out = zeros(T, d, d, K)
    for k in 1:K, (b, idx) in enumerate(plan.block_indices)
        n = length(idx)
        off = plan.packed_offsets[b]
        for (jj, j) in enumerate(idx), (ii, i) in enumerate(idx)
            out[i, j, k] = packed[off+(jj-1)*n+ii, k]
        end
    end
    return out
end
