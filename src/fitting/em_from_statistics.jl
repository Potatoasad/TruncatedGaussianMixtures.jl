#=
EM M-step computed from weighted sufficient statistics only.

Given, for every component k,

    Nk[k]        = Σ_n w_n r_nk
    S1[:, k]     = Σ_n w_n r_nk x_n                   (d × K)
    S2[:, k]     = Σ_n w_n r_nk x_n x_nᵀ, packed blocks (P × K)

this reproduces the legacy `update!` M-step without ever looking at the N
observations:

    wmean = S1/Nk,  wcov = S2/Nk - wmean wmeanᵀ
    μ_new = wmean - m                         m  = E[Y],    Y = X - μ_old
    Σ_new = wcov + m mᵀ + Σ_old - M2          M2 = E[YYᵀ]   (per block)

For diagonal covariances with `moment_matching=true` (the legacy
high-entropy branch) each coordinate is instead fitted by matching the
truncated-normal mean and variance to (wmean, wcov).
=#

using ForwardDiff: Dual, value, partials

struct _MomentMatchTag end

"""
    truncated_normal_mean_var(μ, s, a, b) -> (mean, var)

Mean and variance of `N(μ, s)` truncated to `[a, b]` (generic, ForwardDiff friendly).
"""
function truncated_normal_mean_var(μ, s, a, b)
    σ = sqrt(s)
    α = (a - μ) / σ
    β = (b - μ) / σ
    logZ = log_ndtr_diff(α, β)
    λa = isfinite(α) ? exp(logphi(α) - logZ) : zero(logZ)
    λb = isfinite(β) ? exp(logphi(β) - logZ) : zero(logZ)
    αλa = isfinite(α) ? α * λa : zero(logZ)
    βλb = isfinite(β) ? β * λb : zero(logZ)
    m = μ + σ * (λa - λb)
    v = s * (1 + αλa - βλb - (λa - λb)^2)
    return m, v
end

function _mean_var_jacobian(μ::Float64, s::Float64, a::Float64, b::Float64)
    dμ = Dual{_MomentMatchTag}(μ, 1.0, 0.0)
    ds = Dual{_MomentMatchTag}(s, 0.0, 1.0)
    m, v = truncated_normal_mean_var(dμ, ds, a, b)
    pm = partials(m)
    pv = partials(v)
    return value(m), value(v), pm[1], pm[2], pv[1], pv[2]
end

# the legacy stopping rule of `do_iteration!` (absolute thresholds kept for compatibility)
_legacy_moment_match_stop(μ, s, a, b) =
    !(isfinite(μ) && isfinite(s)) || s > 100 || s < 1e-9 ||
    is_beyond_lim(μ, sqrt(s), a, b; lower_lim=-4, upper_lim=4)

"""
    moment_match_1d(target_mean, target_var, a, b, μ0, s0; maxiter=50, rtol=1e-10)
        -> (μ, s, status)

Find `(μ, s)` such that `N(μ, s)` truncated to `[a, b]` has the target mean and
variance. Newton with analytic (dual-number) 2×2 Jacobian, warm-started from
the previous parameters `(μ0, s0)`, then from the targets. Falls back to the
legacy `iterate_to_conclusion` (and reports it) when Newton does not converge
inside the legacy safe region, so behavior matches the legacy fitter.
"""
function moment_match_1d(tm::Float64, tv::Float64, a::Float64, b::Float64, μ0::Float64, s0::Float64;
                         maxiter::Int=50, rtol::Float64=1e-10)
    # the legacy procedure never leaves the starting point when the targets
    # already trip its stopping rule: it only applies `fix_kernels_until_good`
    if _legacy_moment_match_stop(tm, tv, a, b)
        μf, σf = fix_kernels_until_good(tm, sqrt(max(tv, 0.0)), a, b; lower_lim=-5, upper_lim=5)
        return μf, σf^2, :moment_match_legacy_fix
    end
    mscale = sqrt(tv)
    for (μ, s) in ((μ0, s0), (tm, tv))
        (isfinite(μ) && isfinite(s) && s > 0) || continue
        for _ in 1:maxiter
            m, v, dmdμ, dmds, dvdμ, dvds = _mean_var_jacobian(μ, s, a, b)
            r1 = m - tm
            r2 = v - tv
            if abs(r1) <= rtol * mscale && abs(r2) <= rtol * tv
                _legacy_moment_match_stop(μ, s, a, b) && break
                return μ, s, :ok
            end
            det = dmdμ * dvds - dmds * dvdμ
            (isfinite(det) && det != 0) || break
            Δμ = -(dvds * r1 - dmds * r2) / det
            Δs = -(-dvdμ * r1 + dmdμ * r2) / det
            # damp steps that would leave s > 0
            t = 1.0
            while s + t * Δs <= 0 && t > 1e-10
                t /= 2
            end
            μ += t * Δμ
            s += t * Δs
            _legacy_moment_match_stop(μ, s, a, b) && break
        end
    end
    x = iterate_to_conclusion(tm, tv, a, b)
    return x[1], x[2], :moment_match_legacy_fallback
end

"""
    em_m_step_from_statistics!(ctx, Nk, S1, S2, means, covariances, lower, upper;
                               moment_matching=false, n_data, variance_floor=0.0)

One complete EM M-step for all components in a single call:

1. truncated moments at the old parameters (reused from the previous oracle
   evaluation when the parameters are identical),
2. the component updates (threaded over components),
3. log normalizers and model moments at the *new* parameters, which the next
   E-step (and M-step) will use.

`S2` holds the packed allowed blocks (P × K, see [`MomentOraclePlan`](@ref)).
Returns a NamedTuple with new `means` (d×K), `covariances` (d×d×K),
`mixture_weights`, `log_normalizers`, model moments and per-component `status`.
"""
function em_m_step_from_statistics!(ctx::OracleContext{T}, Nk::AbstractVector, S1::AbstractMatrix,
                                    S2::AbstractMatrix, means::AbstractMatrix, covariances::AbstractArray,
                                    lower::AbstractVector, upper::AbstractVector;
                                    moment_matching::Bool=false, n_data::Integer,
                                    variance_floor::Real=0.0) where {T}
    t0 = time_ns()
    plan = ctx.plan
    d = plan.d
    _copy_inputs!(ctx, means, covariances, lower, upper)
    K = ctx.K
    (length(Nk) == K && size(S1) == (d, K) && size(S2) == (plan.packed_length, K)) ||
        throw(DimensionMismatch("statistics have inconsistent shapes"))
    Nk_ = convert(Vector{T}, Nk)
    S1_ = convert(Matrix{T}, S1)
    S2_ = convert(Matrix{T}, S2)

    # 1. moments at the old parameters
    _evaluate_inputs!(ctx)
    t1 = time_ns()

    old_means = copy(ctx.in_means)
    old_covs = copy(ctx.in_covariances)
    lo = copy(ctx.in_lower)
    hi = copy(ctx.in_upper)
    new_means = copy(old_means)
    new_covs = zeros(T, d, d, K)
    η = zeros(T, K)
    mstatus = fill(:ok, K)
    total = sum(Nk_)
    repair = ((hi .- lo) ./ max(n_data, 1)) .^ 2

    # 2. component updates
    run_chunked!(ctx) do k, ows
        mstatus[k] = _m_step_component!(new_means, new_covs, η, k, ows, ctx, Nk_, S1_, S2_,
                                         old_means, old_covs, lo, hi, total, repair,
                                         moment_matching, T(variance_floor))
    end
    s = sum(η)
    s > 0 && (η ./= s)
    t2 = time_ns()

    # 3. oracle at the new parameters
    copyto!(ctx.in_means, new_means)
    copyto!(ctx.in_covariances, new_covs)
    _evaluate_inputs!(ctx)
    t3 = time_ns()

    status = [mstatus[k] === :ok ? ctx.status[k] : mstatus[k] for k in 1:K]
    nonfatal = (:ok, :empty_component, :moment_match_legacy_fix, :moment_match_legacy_fallback,
             :repaired_nonpositive_covariance)
    return (
        means=new_means,
        covariances=new_covs,
        mixture_weights=η,
        log_normalizers=copy(ctx.log_normalizers),
        model_means=copy(ctx.model_means),
        model_second_moments=copy(ctx.model_second),
        status=String.(status),
        success=all(st -> st in nonfatal, status) && all(==(:ok), ctx.status),
        elapsed_old_moments=(t1 - t0) / 1e9,
        elapsed_update=(t2 - t1) / 1e9,
        elapsed_new_moments=(t3 - t2) / 1e9,
    )
end

function _m_step_component!(new_means, new_covs, η, k, ows, ctx::OracleContext{T}, Nk, S1, S2,
                            old_means, old_covs, lo, hi, total, repair, moment_matching,
                            variance_floor) where {T}
    plan = ctx.plan
    d = plan.d
    nk = Nk[k]
    # keep old parameters for components that received no mass
    if !(nk > 0 && isfinite(nk))
        new_covs[:, :, k] .= @view old_covs[:, :, k]
        η[k] = zero(T)
        return :empty_component
    end
    if ctx.status[k] !== :ok
        new_covs[:, :, k] .= @view old_covs[:, :, k]
        η[k] = nk / total
        return Symbol(:old_parameters_invalid_, ctx.status[k])
    end
    η[k] = nk / total
    status = :ok
    @inbounds for (b, idx) in enumerate(plan.block_indices)
        n = length(idx)
        off = plan.packed_offsets[b]
        for (jj, j) in enumerate(idx)
            wm_j = S1[j, k] / nk
            for (ii, i) in enumerate(idx)
                wm_i = S1[i, k] / nk
                p = off + (jj - 1) * n + ii
                wcov = S2[p, k] / nk - wm_i * wm_j
                m_i = ctx.centered_means[i, k]
                m_j = ctx.centered_means[j, k]
                new_covs[i, j, k] = wcov + m_i * m_j + old_covs[i, j, k] - ctx.centered_second[p, k]
            end
            new_means[j, k] = wm_j - ctx.centered_means[j, k]
        end
        if moment_matching && n == 1
            j = idx[1]
            tm = S1[j, k] / nk
            tv = S2[off+1, k] / nk - tm^2
            μ, s, st = moment_match_1d(tm, tv, lo[j], hi[j], old_means[j, k], old_covs[j, j, k])
            new_means[j, k] = μ
            new_covs[j, j, k] = s
            st === :ok || (status = st)
        end
        # symmetrize the block
        for (jj, j) in enumerate(idx), (ii, i) in enumerate(idx)
            ii > jj || continue
            s = (new_covs[i, j, k] + new_covs[j, i, k]) / 2
            new_covs[i, j, k] = s
            new_covs[j, i, k] = s
        end
    end
    if variance_floor > 0
        for i in 1:d
            new_covs[i, i, k] = max(new_covs[i, i, k], variance_floor)
        end
    end
    if !_all_blocks_pd(new_covs, k, ows, plan)
        # legacy repair: drop the component weight and inflate the variances
        η[k] = zero(T)
        for i in 1:d
            new_covs[i, i, k] += repair[i]
        end
        if _all_blocks_pd(new_covs, k, ows, plan)
            return :repaired_nonpositive_covariance
        end
        new_means[:, k] .= @view old_means[:, k]
        new_covs[:, :, k] .= @view old_covs[:, :, k]
        return :failed_nonpositive_covariance
    end
    return status
end

function _all_blocks_pd(covs, k, ows::OracleWorkspace, plan::MomentOraclePlan)
    @inbounds for (b, idx) in enumerate(plan.block_indices)
        ws = ows.blocks[b]
        n = length(idx)
        for (jj, j) in enumerate(idx), (ii, i) in enumerate(idx)
            ws.Σ[ii, jj] = covs[i, j, k]
        end
        small_cholesky!(ws.chol, ws.Σ, n) || return false
    end
    return true
end

"""
    em_m_step_from_statistics(Nk, S1, S2, means, covariances, lower, upper, block_labels=nothing;
                              covariance_type=:full, kwargs...)

Allocating convenience wrapper (builds a temporary [`OracleContext`](@ref)).
"""
function em_m_step_from_statistics(Nk, S1, S2, means, covariances, lower, upper, block_labels=nothing;
                                   covariance_type=:full, threaded=true, specialize=true, seed=0, kwargs...)
    d, K = size(means)
    ctx = OracleContext(d, K, block_labels, covariance_type; threaded=threaded, specialize=specialize, seed=seed)
    em_m_step_from_statistics!(ctx, Nk, S1, S2, means, covariances, lower, upper; kwargs...)
end
