#usininclude(joinpath(@__DIR__, "../src/TruncatedGaussianMixtures.jl"))
using TruncatedGaussianMixtures
using Distributions, LinearAlgebra, Test

function generate_random_mixture(;d = 2, K=3, cov=:full)
	scaling = rand(0.5:0.1:10.0, d);
	diag_scaling = rand(0.5:0.1:2.0)	
	a = zeros(d)
	b = scaling
	
	rand_rotation = qr(randn(d,d)).Q
	
	weights = rand(Dirichlet(30.0.*ones(K)))
	comps = TruncatedMvNormal[]
	for k ∈ 1:K
		μ = (b .- a).*rand(d) .+ a;
		if cov == :diag
			rand_rotation = one(rand_rotation)
		end
		Σ = rand_rotation' * diagm(rand(d).*diag_scaling.*0.1) * rand_rotation;
		Σ .= (Σ + Σ')/2
		push!(comps, TruncatedMvNormal(MvNormal(μ,Σ),a,b))
	end
	
	MixtureModel(comps, weights)
end

function KLDivergence(d1, d2; N_samps=8_000)
	N_samps = length(d1)*N_samps
	X = rand(d1, N_samps); Y = rand(d2, N_samps)
	k1 = mean([loglikelihood(d1, X[:,i]) - loglikelihood(d2, X[:,i]) for i ∈ 1:N_samps])
	k2 = mean([loglikelihood(d2, Y[:,i]) - loglikelihood(d1, Y[:,i]) for i ∈ 1:N_samps])
	0.5*(k1+k2)
end

function test_result_pair(;d=2, K=3, cov=:full, tol=1e-2, MAX_REPS=100, verbose=false, progress=false)
	test_mixture = generate_random_mixture(d=d, K=K, cov=cov);
	fitted_mixture = fit_gmm(rand(test_mixture,8000),
							length(test_mixture.components),
							test_mixture.components[1].a,
							test_mixture.components[1].b,
							cov=cov, verbose=verbose, tol=tol,MAX_REPS=MAX_REPS, progress=progress)
	test_mixture, fitted_mixture
end

@testset "Fitting Randomly generated Mixture Models" begin
	@test abs(KLDivergence(test_result_pair(d=2, K=3, cov=:full, tol=1e-3)...)) ≤ 2e-2
	@test abs(KLDivergence(test_result_pair(d=2, K=3, cov=:diag, tol=1e-3)...)) ≤ 1e-2
	@test abs(KLDivergence(test_result_pair(d=2, K=20, cov=:full, tol=1e-3)...)) ≤ 2e-2
	@test abs(KLDivergence(test_result_pair(d=2, K=20, cov=:diag, tol=1e-3)...)) ≤ 1e-2
	@test abs(KLDivergence(test_result_pair(d=10, K=10, cov=:full, tol=1e-3)...)) ≤ 2e-2
	@test abs(KLDivergence(test_result_pair(d=10, K=10, cov=:diag, tol=1e-3)...)) ≤ 1e-2
end