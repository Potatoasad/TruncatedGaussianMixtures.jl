#=
Precompile workload for the oracle / M-step, so that the first fit in a fresh
(Python or Julia) session does not pay for compiling them.
=#
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    layouts = ((:diag, nothing, 3), (:full, [1, 1, 2, 2, 2, 3, 3, 3, 3], 9))
    @compile_workload begin
        for (cov, labels, d) in layouts
            K = 3
            μ = zeros(d, K)
            Σ = zeros(d, d, K)
            for k in 1:K, i in 1:d
                μ[i, k] = 0.1 * k - 0.05 * i
                Σ[i, i, k] = 0.5 + 0.1 * i
            end
            if labels !== nothing
                for k in 1:K, i in 1:d, j in 1:d
                    if i != j && labels[i] == labels[j]
                        Σ[i, j, k] = 0.05
                    end
                end
            end
            lo = fill(-1.0, d)
            hi = fill(1.0, d)
            # legacy k-means initializer (used by both Python backends)
            X = 1.8 .* rand(Xoshiro(1), d, 200) .- 0.9
            initialize(X, K, lo, hi; cov=cov)
            initialize(X, K, lo, hi; cov=cov, weights=StatsBase.Weights(fill(1.0, 200)))
            for threaded in (false, true)
                ctx = OracleContext(d, K, labels, cov; threaded=threaded)
                r = component_oracle!(ctx, μ, Σ, lo, hi)
                P = ctx.plan.packed_length
                Nk = fill(10.0, K)
                S1 = Nk' .* r.model_means
                S2 = Nk' .* r.model_second_moments
                em_m_step_from_statistics!(ctx, Nk, S1, S2, μ, Σ, lo, hi; moment_matching=false, n_data=30)
                if cov == :diag
                    em_m_step_from_statistics!(ctx, Nk, S1, S2, μ, Σ, lo, hi; moment_matching=true, n_data=30)
                end
            end
        end
    end
end
