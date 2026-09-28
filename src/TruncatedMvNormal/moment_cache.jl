#=
Cached truncated-normal moment machinery (the "moment oracle").

For a centered Gaussian Y ~ N(0, Σ) truncated to lo < Y < hi (one covariance
block), this file computes

    log Z = log P(lo < Y < hi),   E[Y],   E[Y Yᵀ]

using the same formula as the legacy `EXᵢ!` / `EXᵢXⱼ!`, but with every distinct
boundary quantity evaluated exactly once:

    1                  box probability Z
    2n                 single-face terms  F(k, lo_k), F(k, hi_k)
    4 * binomial(n,2)  pair-corner terms  F_kq(corner)

followed by cheap rank-one contractions

    E[YYᵀ] = Σ + Σ_k a_k c_k c_kᵀ + Σ_k c_k v_kᵀ.

Box probabilities use the closed form in 1D and MvNormalCDF.mvnormcdf for
dimension ≥ 2 (seeded per workspace).
=#

using Random: Xoshiro, seed!
using StatsFuns: normlogcdf, normlogccdf, normccdf

const LOG2π = log(2π)

@inline logphi(z) = -(z * z + LOG2π) / 2

"""
    log_ndtr_diff(a, b)

`log(Φ(b) - Φ(a))` for the standard normal, computed without cancellation in
either tail. Returns `-Inf` when `a ≥ b`.
"""
function log_ndtr_diff(a, b)
    a < b || return oftype(float(a), -Inf)
    if a > 0
        # upper tail: Φc(a) - Φc(b)
        return logsubexp(normlogccdf(a), normlogccdf(b))
    elseif b < 0
        return logsubexp(normlogcdf(b), normlogcdf(a))
    else
        # straddles zero: both Φ(a) and Φc(b) are ≤ 1/2, no cancellation
        return log1p(-normcdf(a) - normccdf(b))
    end
end

"""
    signed_difference_from_logs(logx, logy) -> (sign, log|x - y|)
"""
@inline function signed_difference_from_logs(logx::T, logy::T) where {T}
    if logx == logy
        return (0, T(-Inf))
    elseif logx > logy
        return (1, logsubexp(logx, logy))
    else
        return (-1, logsubexp(logy, logx))
    end
end

"""
    signed_sum_from_logs(logvalues, signs) -> (sign, log|Σ signs[i] * exp(logvalues[i])|)
"""
@inline function signed_sum_from_logs(logvalues::NTuple{N,T}, signs::NTuple{N,Int}) where {N,T}
    m = maximum(logvalues)
    isfinite(m) || return (0, T(-Inf))
    s = zero(T)
    @inbounds for i in 1:N
        s += signs[i] * exp(logvalues[i] - m)
    end
    s == 0 && return (0, T(-Inf))
    return (s > 0 ? 1 : -1, m + log(abs(s)))
end

@inline function _signed_normalized(signed_log::Tuple{Int,T}, logZ::T) where {T}
    sgn, la = signed_log
    sgn == 0 && return zero(T)
    return sgn * exp(la - logZ)
end

############################################################################
# Plans and workspaces
############################################################################

"""
Immutable per-block-size metadata: complements of single indices and of
unordered pairs. Built once per fit.
"""
struct BlockMomentPlan
    size::Int
    complement_of_index::Vector{Vector{Int}}
    pairs::Vector{Tuple{Int,Int}}
    complement_of_pair::Vector{Vector{Int}}
end

function BlockMomentPlan(n::Int)
    n ≥ 1 || throw(ArgumentError("block size must be ≥ 1"))
    comp = [[j for j in 1:n if j != k] for k in 1:n]
    pairs = [(k, q) for k in 1:n for q in (k+1):n]
    comp2 = [[j for j in 1:n if j != k && j != q] for (k, q) in pairs]
    BlockMomentPlan(n, comp, pairs, comp2)
end

"""
Normalized boundary cache shared by `EXᵢ` and `EXᵢXⱼ` for one block evaluation.
"""
mutable struct BlockMomentCache{T}
    logZ::T
    face_lower::Vector{T}       # F(k, lo[k]) / Z
    face_upper::Vector{T}       # F(k, hi[k]) / Z
    face_difference::Vector{T}  # lower - upper
    face_weighted::Vector{T}    # lo .* lower - hi .* upper
    pair_jump::Matrix{T}        # normalized alternating corner sum
end

BlockMomentCache{T}(n::Int) where {T} =
    BlockMomentCache{T}(zero(T), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n, n))

"""
Mutable scratch space for one block size, owned by one task.
"""
mutable struct BlockMomentWorkspace{T}
    n::Int
    cache::BlockMomentCache{T}
    Σ::Matrix{T}        # n×n block covariance
    chol::Matrix{T}     # n×n scratch Cholesky
    lo::Vector{T}       # centered bounds
    hi::Vector{T}
    m1::Vector{T}       # centered E[Y]
    m2::Matrix{T}       # centered E[YYᵀ]
    v::Vector{T}
    # face (single coordinate fixed): n-1 dimensional conditional problem
    A1::Vector{T}
    C1::Matrix{T}
    lo1::Vector{T}
    hi1::Vector{T}
    # pair (two coordinates fixed): n-2 dimensional conditional problem
    A2::Matrix{T}
    C2::Matrix{T}
    lo2::Vector{T}
    hi2::Vector{T}
    # MvNormalCDF settings / diagnostics (dimension ≥ 2 box probabilities)
    rng::Xoshiro
    qmc_points_per_dim::Int
    max_cdf_relative_error::T
    # instrumentation: number of box / face / pair-corner CDF evaluations
    n_box::Int
    n_face::Int
    n_pair::Int
end

function BlockMomentWorkspace{T}(n::Int; qmc_points_per_dim::Int=1000) where {T}
    n1 = max(n - 1, 0)
    n2 = max(n - 2, 0)
    BlockMomentWorkspace{T}(n, BlockMomentCache{T}(n),
        zeros(T, n, n), zeros(T, n, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n, n), zeros(T, n),
        zeros(T, n1), zeros(T, n1, n1), zeros(T, n1), zeros(T, n1),
        zeros(T, n2, 2), zeros(T, n2, n2), zeros(T, n2), zeros(T, n2),
        Xoshiro(0), qmc_points_per_dim, zero(T), 0, 0, 0)
end

reset_counters!(ws::BlockMomentWorkspace) = (ws.n_box = 0; ws.n_face = 0; ws.n_pair = 0; ws)

############################################################################
# Small dense linear algebra (allocation free)
############################################################################

"""
In-place lower Cholesky of the n×n leading block of `A` into `L`.
Returns `false` if `A` is not (numerically) positive definite.
"""
function small_cholesky!(L::AbstractMatrix{T}, A::AbstractMatrix{T}, n::Int) where {T}
    @inbounds for j in 1:n
        s = A[j, j]
        for p in 1:(j-1)
            s -= L[j, p]^2
        end
        (s > 0 && isfinite(s)) || return false
        ljj = sqrt(s)
        L[j, j] = ljj
        for i in (j+1):n
            t = A[i, j]
            for p in 1:(j-1)
                t -= L[i, p] * L[j, p]
            end
            L[i, j] = t / ljj
        end
        for i in 1:(j-1)
            L[i, j] = zero(T)
        end
    end
    return true
end

############################################################################
# Box probabilities of centered Gaussians
############################################################################

"""
    log_box_probability(ws, C, lo, hi, n)

`log P(lo < Y < hi)` for `Y ~ N(0, C)` of dimension `n` (C is n×n).
Dimension 1 uses the closed form; dimension ≥ 2 uses `MvNormalCDF.mvnormcdf`
with the workspace's seeded RNG (so results are reproducible and thread safe).
"""
function log_box_probability(ws::BlockMomentWorkspace{T}, C::AbstractMatrix{T},
                             lo::AbstractVector{T}, hi::AbstractVector{T}, n::Int) where {T}
    if n == 0
        return zero(T)
    elseif n == 1
        σ = sqrt(C[1, 1])
        return log_ndtr_diff(lo[1] / σ, hi[1] / σ)
    else
        p, e = mvnormcdf(C, lo, hi; m=ws.qmc_points_per_dim * n, rng=ws.rng)
        p > 0 || return T(-Inf)
        ws.max_cdf_relative_error = max(ws.max_cdf_relative_error, e / p)
        return log(p)
    end
end

############################################################################
# Boundary cache
############################################################################

"""
    fill_boundary_cache!(ws, plan) -> status::Symbol

Given `ws.Σ`, `ws.lo`, `ws.hi` (centered), fill `ws.cache` with log Z and the
normalized face and pair-corner terms, each evaluated exactly once.
"""
function fill_boundary_cache!(ws::BlockMomentWorkspace{T}, plan::BlockMomentPlan) where {T}
    n = plan.size
    Σ = ws.Σ
    lo = ws.lo
    hi = ws.hi
    cache = ws.cache

    small_cholesky!(ws.chol, Σ, n) || return :nonpositive_definite_covariance

    logZ = log_box_probability(ws, Σ, lo, hi, n)
    ws.n_box += 1
    cache.logZ = logZ
    (isfinite(logZ)) || return :zero_truncation_mass

    # ---- single faces: F(k, x) = φ_{Σkk}(x) · P(box of rest | Y_k = x)
    @inbounds for k in 1:n
        σkk = Σ[k, k]
        sk = sqrt(σkk)
        rest = plan.complement_of_index[k]
        m = n - 1
        for (ii, i) in enumerate(rest)
            ws.A1[ii] = Σ[i, k] / σkk
        end
        for (jj, j) in enumerate(rest), (ii, i) in enumerate(rest)
            ws.C1[ii, jj] = Σ[i, j] - Σ[i, k] * Σ[k, j] / σkk
        end
        L = zero(T)
        U = zero(T)
        wl = zero(T)
        wu = zero(T)
        for side in 1:2
            x = side == 1 ? lo[k] : hi[k]
            isfinite(x) || continue
            for (ii, i) in enumerate(rest)
                ws.lo1[ii] = lo[i] - ws.A1[ii] * x
                ws.hi1[ii] = hi[i] - ws.A1[ii] * x
            end
            logF = logphi(x / sk) - log(sk) + log_box_probability(ws, ws.C1, ws.lo1, ws.hi1, m)
            ws.n_face += 1
            val = exp(logF - logZ)
            if side == 1
                L = val
                wl = x * val
            else
                U = val
                wu = x * val
            end
        end
        cache.face_lower[k] = L
        cache.face_upper[k] = U
        cache.face_difference[k] = L - U
        cache.face_weighted[k] = wl - wu
    end

    # ---- pair corners: F_kq(x) = φ_2(x; Σ_{kq,kq}) · P(box of rest | Y_k, Y_q = x)
    fill!(cache.pair_jump, zero(T))
    @inbounds for (pidx, (k, q)) in enumerate(plan.pairs)
        skk = Σ[k, k]
        sqq = Σ[q, q]
        skq = Σ[k, q]
        det2 = skk * sqq - skq * skq
        det2 > 0 || return :nonpositive_definite_covariance
        rest = plan.complement_of_pair[pidx]
        m = n - 2
        # A = Σ[rest, (k,q)] * inv(S2), with the closed-form 2×2 SPD solve
        for (ii, i) in enumerate(rest)
            sik = Σ[i, k]
            siq = Σ[i, q]
            ws.A2[ii, 1] = (sik * sqq - siq * skq) / det2
            ws.A2[ii, 2] = (siq * skk - sik * skq) / det2
        end
        for (jj, j) in enumerate(rest), (ii, i) in enumerate(rest)
            ws.C2[ii, jj] = Σ[i, j] - (ws.A2[ii, 1] * Σ[k, j] + ws.A2[ii, 2] * Σ[q, j])
        end
        logdet2 = log(det2)
        l1 = _pair_corner_logF(ws, Σ, lo, hi, rest, m, lo[k], lo[q], skk, sqq, skq, det2, logdet2)
        l2 = _pair_corner_logF(ws, Σ, lo, hi, rest, m, hi[k], hi[q], skk, sqq, skq, det2, logdet2)
        l3 = _pair_corner_logF(ws, Σ, lo, hi, rest, m, lo[k], hi[q], skk, sqq, skq, det2, logdet2)
        l4 = _pair_corner_logF(ws, Σ, lo, hi, rest, m, hi[k], lo[q], skk, sqq, skq, det2, logdet2)
        jump = _signed_normalized(signed_sum_from_logs((l1, l2, l3, l4), (1, 1, -1, -1)), logZ)
        cache.pair_jump[k, q] = jump
        cache.pair_jump[q, k] = jump
    end
    return :ok
end

# log F_kq at corner (xk, xq): bivariate marginal density times the conditional
# box probability of the remaining coordinates (uses ws.A2 / ws.C2).
@inline function _pair_corner_logF(ws::BlockMomentWorkspace{T}, Σ, lo, hi, rest, m::Int,
                                   xk::T, xq::T, skk::T, sqq::T, skq::T, det2::T, logdet2::T) where {T}
    (isfinite(xk) && isfinite(xq)) || return T(-Inf)
    quad = (sqq * xk * xk - 2 * skq * xk * xq + skk * xq * xq) / det2
    logφ2 = -LOG2π - logdet2 / 2 - quad / 2
    @inbounds for (ii, i) in enumerate(rest)
        shift = ws.A2[ii, 1] * xk + ws.A2[ii, 2] * xq
        ws.lo2[ii] = lo[i] - shift
        ws.hi2[ii] = hi[i] - shift
    end
    ws.n_pair += 1
    return logφ2 + log_box_probability(ws, ws.C2, ws.lo2, ws.hi2, m)
end

############################################################################
# Centered moments from the cache
############################################################################

"""
    centered_moments_general!(ws, plan) -> status

General cached path for any block size. Reads `ws.Σ`, `ws.lo`, `ws.hi`
(bounds already shifted by the mean) and writes log Z to `ws.cache.logZ`,
E[Y] to `ws.m1` and E[YYᵀ] to `ws.m2`.
"""
function centered_moments_general!(ws::BlockMomentWorkspace{T}, plan::BlockMomentPlan) where {T}
    status = fill_boundary_cache!(ws, plan)
    status === :ok || return status
    n = plan.size
    Σ = ws.Σ
    cache = ws.cache
    m1 = ws.m1
    m2 = ws.m2
    v = ws.v
    @inbounds begin
        # E[Y] = Σ h
        for i in 1:n
            s = zero(T)
            for k in 1:n
                s += Σ[i, k] * cache.face_difference[k]
            end
            m1[i] = s
        end
        copyto!(m2, Σ)
        for k in 1:n
            σkk = Σ[k, k]
            ak = cache.face_weighted[k] / σkk
            # v_k[j] = Σ_{q≠k} (Σ[j,q] - Σ[k,q]Σ[j,k]/σkk) P[k,q]
            for j in 1:n
                v[j] = zero(T)
            end
            for q in 1:n
                q == k && continue
                pkq = cache.pair_jump[k, q]
                pkq == 0 && continue
                for j in 1:n
                    v[j] += (Σ[j, q] - Σ[k, q] * Σ[j, k] / σkk) * pkq
                end
            end
            for j in 1:n, i in 1:n
                m2[i, j] += ak * Σ[i, k] * Σ[j, k] + Σ[i, k] * v[j]
            end
        end
        # symmetrize
        for j in 1:n, i in (j+1):n
            s = (m2[i, j] + m2[j, i]) / 2
            m2[i, j] = s
            m2[j, i] = s
        end
    end
    return _check_moments(ws, n)
end

function _check_moments(ws::BlockMomentWorkspace{T}, n::Int) where {T}
    @inbounds for i in 1:n
        isfinite(ws.m1[i]) || return :nonfinite_moments
        for j in 1:n
            isfinite(ws.m2[i, j]) || return :nonfinite_moments
        end
    end
    # covariance M2 - m m' must be PSD (up to a small relative tolerance)
    @inbounds for j in 1:n, i in 1:n
        ws.chol[i, j] = ws.m2[i, j] - ws.m1[i] * ws.m1[j]
    end
    @inbounds for i in 1:n
        ws.chol[i, i] += 1e-10 * abs(ws.Σ[i, i])
    end
    # reuse ws.C1-independent storage: factor in place (copy needed since input==output)
    ok = _psd_check_inplace!(ws.chol, n)
    ok || return :nonpsd_moment_covariance
    return :ok
end

function _psd_check_inplace!(A::AbstractMatrix{T}, n::Int) where {T}
    @inbounds for j in 1:n
        s = A[j, j]
        for p in 1:(j-1)
            s -= A[j, p]^2
        end
        s > 0 || return false
        ljj = sqrt(s)
        A[j, j] = ljj
        for i in (j+1):n
            t = A[i, j]
            for p in 1:(j-1)
                t -= A[i, p] * A[j, p]
            end
            A[i, j] = t / ljj
        end
    end
    return true
end
