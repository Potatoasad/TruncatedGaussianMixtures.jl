using Plots, Distributions, LinearAlgebra, Test
using Revise, TruncatedGaussianMixtures
using PairPlots
using DataFrames
using CairoMakie

function generate_random_mixture(;d = 2, K=3, cov=:full)
	scaling = rand(0.5:0.1:10.0, d);
	diag_scaling = rand(0.5:0.1:6.0)	
	a = zeros(d)
	b = scaling
	
	rand_rotation = qr(randn(d,d)).R
	
	weights = rand(Dirichlet(30.0.*ones(K)))
	comps = TruncatedMvNormal[]
	for k ∈ 1:K
		μ = (b .- a).*rand(d) .+ a;
		if cov == :diag
			rand_rotation = one(rand_rotation)
		end
		Σ = rand_rotation' * diagm(rand(d).*diag_scaling.*1.0) * rand_rotation;
		Σ .= (Σ + Σ')/2
		print(Σ)
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

function test_result_pair(;d=2, K=3, cov=:full, tol=1e-6, MAX_REPS=500, verbose=false, progress=false)
	test_mixture = generate_random_mixture(d=d, K=K, cov=cov);
	fitted_mixture = fit_gmm(rand(test_mixture,8_000),
							length(test_mixture.components),
							test_mixture.components[1].a,
							test_mixture.components[1].b,
							cov=cov, verbose=verbose, tol=tol,MAX_REPS=MAX_REPS, progress=progress)
	test_mixture, fitted_mixture
end

function test_result_pair2(;d=2, K=3, cov=:full, tol=1e-6, MAX_REPS=500, verbose=false, progress=false)
	test_mixture = generate_random_mixture(d=d, K=K, cov=cov);
	fitted_mixture = fit_gmm(rand(test_mixture,800),
							40,
							test_mixture.components[1].a,
							test_mixture.components[1].b,
							cov=cov, verbose=verbose, tol=tol,MAX_REPS=MAX_REPS, progress=progress)
	test_mixture, fitted_mixture
end

create_df(test::Distribution; N=8_000) = DataFrame(rand(test, N)', [Symbol("x_$(i)") for i ∈ 1:length(test)])


test, fit = test_result_pair(d=7,K=3,cov=:diag, progress=true, tol=1e-8, MAX_REPS=1000)

pp = Plots.plot()
Plots.scatter!(pp, (let x=rand(test, 8000); x[1,:], x[2,:] end)..., alpha=0.1, markerstrokealpha=0.0, color=:red, label="original")
Plots.scatter!(pp, (let x=rand(fit, 8000); x[1,:], x[2,:] end)..., alpha=0.1, markerstrokealpha=0.0, color=:blue, label="fitted")
pp


pairplot(create_df(test), create_df(fit))

