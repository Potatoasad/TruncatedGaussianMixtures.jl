#=
Tests for the cached / batched moment oracle and the M-step from sufficient
statistics. Run standalone with

    julia --project=<env with TGMM, Test, QuadGK> -t 8 test/test_moment_oracle.jl
=#
using TruncatedGaussianMixtures, LinearAlgebra, Random, Test, Distributions
using QuadGK
using SpecialFunctions: erfc
const TGMM = TruncatedGaussianMixtures

bigΦ(x) = erfc(-BigFloat(x) / sqrt(BigFloat(2))) / 2

random_spd(n; scale=1.0, jitter=0.3) = (A = randn(n, n); scale .* (A * A' / n + jitter * I))

function random_corr_spd(n, ρ)
    C = fill(ρ, n, n); C[diagind(C)] .= 1.0
    s = 0.5 .+ rand(n)
    Diagonal(s) * C * Diagonal(s)
end

# Reference centered moments by nested adaptive quadrature (n ≤ 2).
function reference_moments(Σ, lo, hi)
    n = length(lo)
    if n == 1
        d = truncated(Normal(0, sqrt(Σ[1, 1])), lo[1], hi[1])
        Z = cdf(Normal(0, sqrt(Σ[1, 1])), hi[1]) - cdf(Normal(0, sqrt(Σ[1, 1])), lo[1])
        return log(Z), [mean(d)], fill(var(d) + mean(d)^2, 1, 1)
    end
    @assert n == 2
    D = MvNormal(zeros(2), Σ)
    f(g) = quadgk(x -> quadgk(y -> pdf(D, [x, y]) * g(x, y), lo[2], hi[2]; rtol=1e-12)[1], lo[1], hi[1]; rtol=1e-12)[1]
    Z = f((x, y) -> 1.0)
    m = [f((x, y) -> x), f((x, y) -> y)] ./ Z
    M = [f((x, y) -> x * x) f((x, y) -> x * y); f((x, y) -> x * y) f((x, y) -> y * y)] ./ Z
    return log(Z), m, M
end

@testset "moment oracle" begin
    Random.seed!(20260927)

    @testset "stable log helpers" begin
        for (a, b) in ((-1.0, 2.0), (3.0, 4.0), (8.0, 9.0), (-12.0, -11.0), (-Inf, -30.0), (30.0, Inf), (-0.1, 0.1))
            ref = a > 0 ? log(bigΦ(-a) - bigΦ(-b)) : log(bigΦ(b) - bigΦ(a))
            @test TGMM.log_ndtr_diff(a, b) ≈ Float64(ref) rtol = 1e-12
        end
        @test TGMM.log_ndtr_diff(1.0, 1.0) == -Inf
        s, l = TGMM.signed_sum_from_logs((log(3.0), log(1.0), log(2.5), log(0.5)), (1, 1, -1, -1))
        @test s * exp(l) ≈ 1.0
        s, l = TGMM.signed_difference_from_logs(log(1.0), log(3.0))
        @test s == -1 && exp(l) ≈ 2.0
    end

    @testset "reference accuracy (n = 1, 2)" begin
        for n in 1:2, trial in 1:4
            Σ = random_spd(n)
            lo = -0.2 .- 1.5rand(n); hi = 0.3 .+ 1.5rand(n)
            logZ, m1, m2, st = truncated_centered_moments(Σ, lo, hi; qmc_points_per_dim=100_000)
            rlogZ, rm1, rm2 = reference_moments(Σ, lo, hi)
            @test st == :ok
            tol = n == 1 ? 1e-12 : 2e-4
            @test logZ ≈ rlogZ atol = tol
            @test m1 ≈ rm1 atol = tol * sqrt(maximum(diag(Σ)))
            @test m2 ≈ rm2 atol = tol * maximum(diag(Σ))
        end
    end

    @testset "legacy equivalence (n = 1..5)" begin
        for n in 1:5, trial in 1:3
            Σ = random_spd(n)
            lo = -0.3 .- rand(n); hi = 0.3 .+ rand(n)
            logZ, m1, m2, st = truncated_centered_moments(Σ, lo, hi; qmc_points_per_dim=20_000)
            @test st == :ok
            leg1 = TGMM.EXᵢ(Σ, lo, hi)
            leg2 = TGMM.EXᵢXⱼ(Σ, lo, hi)
            legZ = log(TGMM.α(Σ, lo, hi))
            # the legacy routine uses randomized QMC, so agreement is at its noise level
            s = maximum(diag(Σ))
            @test logZ ≈ legZ atol = 1e-2
            @test m1 ≈ leg1 atol = 1e-2 * sqrt(s)
            @test m2 ≈ leg2 atol = 1e-2 * s
        end
    end

    @testset "specialization equivalence" begin
        for n in 1:3, trial in 1:5
            Σ = random_spd(n)
            lo = -0.3 .- 2rand(n); hi = 0.3 .+ 2rand(n)
            a = truncated_centered_moments(Σ, lo, hi; specialize=true)
            b = truncated_centered_moments(Σ, lo, hi; specialize=false)
            @test a[4] == b[4] == :ok
            @test a[1] ≈ b[1] atol = 1e-12
            @test a[2] ≈ b[2] atol = 1e-12
            @test a[3] ≈ b[3] atol = 1e-12
        end
    end

    @testset "symmetry, PSD, tails" begin
        cases = Any[]
        for n in 1:4
            push!(cases, (random_spd(n), fill(-1.0, n), fill(1.0, n)))                      # ordinary
            push!(cases, (random_spd(n), fill(2.0, n), fill(4.0, n)))                        # box in the tail
            push!(cases, (random_spd(n; scale=1e-2), fill(-0.1, n), fill(0.4, n)))           # narrow variance
            push!(cases, (random_spd(n; scale=1e2), fill(-1.0, n), fill(0.5, n)))            # wide variance
            push!(cases, (random_spd(n), (n == 1 ? [-3.0] : collect(range(-3, 0.5, length=n))), fill(Inf, n)))   # half-open box
            n ≥ 2 && push!(cases, (random_corr_spd(n, 0.9), fill(-0.5, n), fill(1.5, n)))     # strong correlation
            n ≥ 2 && push!(cases, (random_corr_spd(n, -0.9 / (n - 1)), fill(-0.5, n), fill(1.5, n)))
        end
        for (Σ, lo, hi) in cases
            # extreme-tail boxes need more QMC points than the legacy default for
            # relative accuracy; at the default they must be flagged, never silently wrong
            st0 = truncated_centered_moments(Σ, lo, hi)[4]
            @test st0 in (:ok, :nonpsd_moment_covariance)
            logZ, m1, m2, st = truncated_centered_moments(Σ, lo, hi; qmc_points_per_dim=20_000)
            @test st == :ok
            @test isfinite(logZ)
            @test m2 ≈ m2'
            @test minimum(eigvals(Symmetric(m2 - m1 * m1'))) > -1e-8 * maximum(diag(Σ))
            # all mass inside the box
            @test all(lo .- 1e-8 .<= m1 .<= hi .+ 1e-8)
        end
        # 1D deep tail against Distributions
        for (μ, σ) in ((6.0, 1.0), (-7.0, 0.5), (0.5, 0.01))
            logZ, m1, m2, st = truncated_centered_moments(fill(σ^2, 1, 1), [0.0 - μ], [1.0 - μ])
            d = truncated(Normal(μ, σ), 0.0, 1.0)
            @test st == :ok
            @test m1[1] + μ ≈ mean(d) rtol = 1e-7
            @test m2[1] - m1[1]^2 ≈ var(d) rtol = 1e-5
        end
    end

    @testset "explicit failure statuses" begin
        @test truncated_centered_moments([1.0 2.0; 2.0 1.0], [-1.0, -1.0], [1.0, 1.0])[4] == :nonpositive_definite_covariance
        @test truncated_centered_moments(fill(1.0, 1, 1), [1.0], [1.0])[4] == :zero_truncation_mass
        # far-tail boxes stay finite in log space
        @test isfinite(truncated_centered_moments(fill(1.0, 1, 1), [60.0], [61.0])[1])
    end

    @testset "finite-difference identities of log Z" begin
        # ∇_μ log Z = Σ⁻¹ E[Y];  ∂ log Z / ∂Σ = ½ Σ⁻¹ (E[YYᵀ] - Σ) Σ⁻¹
        for n in 1:3
            Σ = random_spd(n)
            a = -0.4 .- rand(n); b = 0.4 .+ rand(n)
            μ = 0.3 .* randn(n)
            logZ(μ, Σ) = truncated_centered_moments(Σ, a .- μ, b .- μ; seed=7, qmc_points_per_dim=50_000)[1]
            _, m1, m2, _ = truncated_centered_moments(Σ, a .- μ, b .- μ; seed=7, qmc_points_per_dim=50_000)
            h = 1e-4
            g = [(logZ(μ .+ h .* (1:n .== i), Σ) - logZ(μ .- h .* (1:n .== i), Σ)) / 2h for i in 1:n]
            tol = n == 1 ? 1e-7 : 2e-3
            @test g ≈ Σ \ m1 atol = tol
            G = 0.5 * (Σ \ (m2 - Σ) / Σ)
            for i in 1:n
                E = zeros(n, n); E[i, i] = 1.0
                gΣ = (logZ(μ, Σ + h * E) - logZ(μ, Σ - h * E)) / 2h
                @test gΣ ≈ G[i, i] atol = tol
            end
        end
    end

    @testset "batched oracle: blocks, determinism, counts, allocations" begin
        d, K = 7, 25
        labels = [3, 3, 1, 5, 5, 5, 9]
        blocks = TGMM.canonical_blocks(labels)
        @test blocks == [[1, 2], [3], [4, 5, 6], [7]]
        μ = randn(d, K)
        Σs = zeros(d, d, K)
        for k in 1:K, b in blocks
            Σs[b, b, k] = random_spd(length(b))
        end
        lo = fill(-1.5, d); hi = fill(2.0, d)
        ctx = OracleContext(d, K, labels, :full; threaded=true)
        r = component_oracle!(ctx, μ, Σs, lo, hi)
        @test r.success
        dense_second = TGMM.unpack_blocks(r.model_second_moments, ctx.plan)
        for k in (1, 13, 25)
            logZk = 0.0
            for (bi, b) in enumerate(blocks)
                # same per-(component, block) seed as the oracle
                ws = TGMM.BlockMomentWorkspace{Float64}(length(b))
                Random.seed!(ws.rng, hash((ctx.seed, k, bi)))
                ws.Σ .= Σs[b, b, k]; ws.lo .= lo[b] .- μ[b, k]; ws.hi .= hi[b] .- μ[b, k]
                @test TGMM.centered_moments!(ws, TGMM.BlockMomentPlan(length(b))) == :ok
                logZk += ws.cache.logZ
                @test r.model_means[b, k] ≈ ws.m1 .+ μ[b, k]
                @test dense_second[b, b, k] ≈ ws.m2 .+ μ[b, k] * ws.m1' .+ ws.m1 * μ[b, k]' .+ μ[b, k] * μ[b, k]'
            end
            @test r.log_normalizers[k] ≈ logZk
        end
        # serial and threaded evaluations agree exactly
        ctx_serial = OracleContext(d, K, labels, :full; threaded=false)
        rs = component_oracle!(ctx_serial, μ, Σs, lo, hi)
        @test rs.log_normalizers == r.log_normalizers
        @test rs.model_second_moments == r.model_second_moments
        # a repeated call with identical parameters reuses the cache
        n0 = ctx.n_evaluations
        component_oracle!(ctx, μ, Σs, lo, hi)
        @test ctx.n_evaluations == n0
        # each distinct boundary term is evaluated exactly once per block
        for n in 1:5
            ws = TGMM.BlockMomentWorkspace{Float64}(n)
            ws.Σ .= random_spd(n); ws.lo .= -1.0; ws.hi .= 1.0
            TGMM.reset_counters!(ws)
            TGMM.centered_moments!(ws, TGMM.BlockMomentPlan(n); specialize=false)
            @test (ws.n_box, ws.n_face, ws.n_pair) == (1, 2n, 4 * binomial(n, 2))
        end
        # the diagonal (size-1) path does not allocate after warm-up
        ctxd = OracleContext(d, K, nothing, :diag; threaded=false)
        Σd = zeros(d, d, K); for k in 1:K; Σd[:, :, k] = Diagonal(0.5 .+ rand(d)); end
        component_oracle!(ctxd, μ, Σd, lo, hi)
        ows = ctxd.workspaces[1]
        TGMM.component_moments!(ctxd, 1, ows, μ, Σd, lo, hi)
        @test (@allocated TGMM.component_moments!(ctxd, 1, ows, μ, Σd, lo, hi)) == 0
    end
end

# Legacy-style Julia E-step (per observation) used as the reference for the
# statistics-based M-step.
function legacy_statistics(X, mix, W, plan)
    N = size(X, 2); K = length(mix.components); d = size(X, 1)
    Nk = zeros(K); S1 = zeros(d, K); S2 = zeros(plan.packed_length, K)
    z = zeros(K)
    for n in 1:N
        TGMM.Zⁿ!(z, mix, X[:, n])
        for k in 1:K
            a = W[n] * z[k]
            Nk[k] += a
            S1[:, k] .+= a .* X[:, n]
            for (b, idx) in enumerate(plan.block_indices)
                off = plan.packed_offsets[b]; nb = length(idx)
                for (jj, j) in enumerate(idx), (ii, i) in enumerate(idx)
                    S2[off+(jj-1)*nb+ii, k] += a * X[i, n] * X[j, n]
                end
            end
        end
    end
    Nk, S1, S2
end

@testset "M-step from statistics vs legacy update!" begin
    Random.seed!(11)
    d, K, N = 3, 4, 3000
    a = zeros(d); b = [1.0, 2.0, 1.5]
    X = rand(d, N) .* b
    for cov in (:diag, :full)
        init = TGMM.initialize(X, K, a, b; cov=cov)
        W = ones(N)
        labels = cov == :full ? [0, 0, 1] : nothing
        plan = TGMM.MomentOraclePlan(d, labels, cov)
        Nk, S1, S2 = legacy_statistics(X, init, W, plan)
        μ = hcat([c.normal.μ for c in init.components]...)
        Σ = cat([Matrix(c.normal.Σ) for c in init.components]...; dims=3)
        if cov == :full
            for k in 1:K; TGMM.project_onto_block!(view(Σ, :, :, k), TGMM.CovarianceBlockStructure(labels)); end
        end
        # legacy
        data = TGMM.StandardData(TGMM.mat2vecs(X))
        EM = ExpectationMaximization(data, init, 1e-6, [zeros(K) for _ in 1:N],
                                     labels === nothing ? zeros(Int, d) : labels, TGMM.StatsBase.Weights(W ./ N))
        mm = cov == :diag && TGMM.normalized_entropy([TGMM.Zⁿ(init, X[:, n]) for n in 1:N]) > 0.5
        update!(EM)
        μleg = hcat([c.normal.μ for c in EM.mix.components]...)
        Σleg = cat([Matrix(c.normal.Σ) for c in EM.mix.components]...; dims=3)
        r = em_m_step_from_statistics(Nk, S1, S2, μ, Σ, a, b, labels; covariance_type=cov,
                                      moment_matching=mm, n_data=N)
        @test r.success
        tol = cov == :diag ? 1e-7 : 5e-3   # the legacy full-covariance moments are QMC-noisy
        @test r.means ≈ μleg atol = tol
        @test r.covariances ≈ Σleg atol = tol
        @test r.mixture_weights ≈ EM.mix.prior.p atol = 1e-10
    end
end

@testset "1D moment matching" begin
    n_ok = 0
    for (tm, tv, a, b) in ((0.4, 0.05, 0.0, 1.0), (0.1, 0.01, 0.0, 1.0), (0.8, 0.03, 0.0, 1.0), (1.2, 0.3, 0.0, 3.0))
        μ, s, st = TGMM.moment_match_1d(tm, tv, a, b, tm, tv)
        leg = TGMM.iterate_to_conclusion(tm, tv, a, b)
        if st == :ok
            n_ok += 1
            m, v = TGMM.truncated_normal_mean_var(μ, s, a, b)
            @test m ≈ tm rtol = 1e-9
            @test v ≈ tv rtol = 1e-9
            @test [μ, s] ≈ leg rtol = 1e-6
        else
            # roots outside the legacy safe region reproduce the legacy result exactly
            @test st == :moment_match_legacy_fallback
            @test [μ, s] == leg
        end
    end
    @test n_ok >= 3
end
