#=
Specialized centered-moment kernels for the block sizes that dominate real
workflows. They produce the same quantities as `centered_moments_general!`
(and are tested against it), but skip the generic complement/pair machinery.

    size 1: closed-form univariate truncated normal
    size 2: faces leave a 1D conditional CDF; pair corners are bare bivariate
            densities; only Z needs MvNormalCDF
    size ≥ 3: general cached path
=#

"""
    centered_moments!(ws, plan; specialize=true) -> status

Fill `ws.cache.logZ`, `ws.m1` (E[Y]) and `ws.m2` (E[YYᵀ]) for the centered
block problem stored in `ws.Σ`, `ws.lo`, `ws.hi`.
"""
function centered_moments!(ws::BlockMomentWorkspace, plan::BlockMomentPlan; specialize::Bool=true)
    if specialize
        n = plan.size
        n == 1 && return centered_moments!(ws, plan, Val(1))
        n == 2 && return centered_moments!(ws, plan, Val(2))
    end
    return centered_moments_general!(ws, plan)
end

function centered_moments!(ws::BlockMomentWorkspace{T}, ::BlockMomentPlan, ::Val{1}) where {T}
    s2 = ws.Σ[1, 1]
    (s2 > 0 && isfinite(s2)) || return :nonpositive_definite_covariance
    σ = sqrt(s2)
    α = ws.lo[1] / σ
    β = ws.hi[1] / σ
    logZ = log_ndtr_diff(α, β)
    ws.n_box += 1
    ws.cache.logZ = logZ
    isfinite(logZ) || return :zero_truncation_mass
    λa = isfinite(α) ? exp(logphi(α) - logZ) : zero(T)
    λb = isfinite(β) ? exp(logphi(β) - logZ) : zero(T)
    ws.n_face += isfinite(α) + isfinite(β)
    αλa = isfinite(α) ? α * λa : zero(T)
    βλb = isfinite(β) ? β * λb : zero(T)
    c = ws.cache
    c.face_lower[1] = λa / σ
    c.face_upper[1] = λb / σ
    c.face_difference[1] = (λa - λb) / σ
    c.face_weighted[1] = αλa - βλb
    ws.m1[1] = σ * (λa - λb)
    ws.m2[1, 1] = s2 * (1 + αλa - βλb)
    return _check_moments(ws, 1)
end

function centered_moments!(ws::BlockMomentWorkspace{T}, ::BlockMomentPlan, ::Val{2}) where {T}
    Σ = ws.Σ
    lo = ws.lo
    hi = ws.hi
    c = ws.cache
    s11, s21, s22 = Σ[1, 1], Σ[2, 1], Σ[2, 2]
    det2 = s11 * s22 - s21 * s21
    (s11 > 0 && s22 > 0 && det2 > 0 && isfinite(det2)) || return :nonpositive_definite_covariance

    logZ = log_box_probability(ws, Σ, lo, hi, 2)
    ws.n_box += 1
    c.logZ = logZ
    isfinite(logZ) || return :zero_truncation_mass

    # single faces: fixing Y_k = x leaves a 1D conditional normal for the other coordinate
    @inbounds for k in 1:2
        o = 3 - k
        skk = Σ[k, k]
        sk = sqrt(skk)
        a = Σ[o, k] / skk
        sc = sqrt(Σ[o, o] - Σ[o, k]^2 / skk)
        L = zero(T); U = zero(T); wl = zero(T); wu = zero(T)
        x = lo[k]
        if isfinite(x)
            L = exp(logphi(x / sk) - log(sk) + log_ndtr_diff((lo[o] - a * x) / sc, (hi[o] - a * x) / sc) - logZ)
            wl = x * L
            ws.n_face += 1
        end
        x = hi[k]
        if isfinite(x)
            U = exp(logphi(x / sk) - log(sk) + log_ndtr_diff((lo[o] - a * x) / sc, (hi[o] - a * x) / sc) - logZ)
            wu = x * U
            ws.n_face += 1
        end
        c.face_lower[k] = L
        c.face_upper[k] = U
        c.face_difference[k] = L - U
        c.face_weighted[k] = wl - wu
    end

    # pair corners: both coordinates fixed, no conditional CDF remains
    logdet2 = log(det2)
    corner(x1, x2) = (isfinite(x1) && isfinite(x2)) ?
        (-LOG2π - logdet2 / 2 - (s22 * x1 * x1 - 2 * s21 * x1 * x2 + s11 * x2 * x2) / (2 * det2)) : T(-Inf)
    l1 = corner(lo[1], lo[2]); l2 = corner(hi[1], hi[2])
    l3 = corner(lo[1], hi[2]); l4 = corner(hi[1], lo[2])
    ws.n_pair += isfinite(l1) + isfinite(l2) + isfinite(l3) + isfinite(l4)
    P = _signed_normalized(signed_sum_from_logs((l1, l2, l3, l4), (1, 1, -1, -1)), logZ)
    c.pair_jump[1, 1] = zero(T); c.pair_jump[2, 2] = zero(T)
    c.pair_jump[1, 2] = P; c.pair_jump[2, 1] = P

    # contractions (n = 2 unrolled)
    h1, h2 = c.face_difference[1], c.face_difference[2]
    ws.m1[1] = s11 * h1 + s21 * h2
    ws.m1[2] = s21 * h1 + s22 * h2
    m11, m21, m22 = s11, s21, s22
    # k = 1: c_1 = (s11, s21), a_1 = w1/s11, v_1[j] = (Σ[j,2] - s12 Σ[j,1]/s11) P
    a1 = c.face_weighted[1] / s11
    v11 = zero(T)                      # j = 1: s12 - s12*s11/s11 = 0
    v12 = (s22 - s21 * s21 / s11) * P  # j = 2
    # k = 2: c_2 = (s21, s22), a_2 = w2/s22, v_2[j] = (Σ[j,1] - s21 Σ[j,2]/s22) P
    a2 = c.face_weighted[2] / s22
    v21 = (s11 - s21 * s21 / s22) * P
    v22 = zero(T)
    m11 += a1 * s11 * s11 + s11 * v11 + a2 * s21 * s21 + s21 * v21
    m22 += a1 * s21 * s21 + s21 * v12 + a2 * s22 * s22 + s22 * v22
    m12 = s21 + a1 * s11 * s21 + s11 * v12 + a2 * s21 * s22 + s21 * v22
    m21 += a1 * s21 * s11 + s21 * v11 + a2 * s22 * s21 + s22 * v21
    off = (m12 + m21) / 2
    ws.m2[1, 1] = m11; ws.m2[2, 2] = m22
    ws.m2[1, 2] = off; ws.m2[2, 1] = off
    return _check_moments(ws, 2)
end

############################################################################
# Convenience (allocating) reference API
############################################################################

"""
    truncated_centered_moments(Σ, lo, hi; specialize=true, seed=0) -> (logZ, m1, m2, status)

Centered moments of `N(0, Σ)` truncated to `lo < Y < hi` via the cached oracle.
"""
function truncated_centered_moments(Σ::AbstractMatrix, lo::AbstractVector, hi::AbstractVector;
                                    specialize::Bool=true, seed::Integer=0, qmc_points_per_dim::Int=1000)
    n = length(lo)
    ws = BlockMomentWorkspace{Float64}(n; qmc_points_per_dim=qmc_points_per_dim)
    seed!(ws.rng, seed)
    copyto!(ws.Σ, Σ); copyto!(ws.lo, lo); copyto!(ws.hi, hi)
    status = centered_moments!(ws, BlockMomentPlan(n); specialize=specialize)
    return ws.cache.logZ, copy(ws.m1), copy(ws.m2), status
end
