using Plots, Distributions, LinearAlgebra, Test
using Revise, TruncatedGaussianMixtures
using PairPlots
using DataFrames
using StatsBase: Weights
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

create_df(test::Distribution; N=8_000) = DataFrame(rand(test, N)', [Symbol("x_$(i)") for i ∈ 1:length(test)])

function generate_weighted_samples(mix, func; N=8_000)
	df = create_df(mix; N=N)
	original_names = names(df) 
	select!(df, :, original_names => ByRow(func) => :weight)
	df
end

function importance_sample(df, N)
	selected_indices = sample(1:nrow(df), Weights(df[!, :weight]), N)
	non_weight_columns = [col for col in Symbol.(names(df)) if col != :weight]
	df[selected_indices, non_weight_columns]
end


function get_original_and_weighted_df(mix, func; N=8_000)
	df = generate_weighted_samples(mix, func; N=N)
	df_original = importance_sample(generate_weighted_samples(mix, func; N=N*50), N)
	df_original, df
end

weight_func = ((x...) -> x[2]^2)

function test_result_pair(;d=2, K=3, cov=:full, tol=1e-6, MAX_REPS=500, verbose=false, progress=false)
	test_mixture = generate_random_mixture(d=d, K=K, cov=cov);
	df = generate_weighted_samples(test_mixture, weight_func;N=8_000)
	non_weight_columns = [col for col in names(df) if col != "weight"]
	(samples, weights) = (collect(Matrix(df[!, non_weight_columns])'),  df[!, :weight])
	fitted_mixture = fit_gmm(samples,
							length(test_mixture.components),
							test_mixture.components[1].a,
							test_mixture.components[1].b,
							cov=cov, verbose=verbose, tol=tol,MAX_REPS=MAX_REPS, progress=progress, weights=Weights(weights))
	test_mixture, fitted_mixture
end


test, fit = test_result_pair(d=2,K=2,cov=:full, progress=false, tol=1e-8, MAX_REPS=100)

#pp = Plots.plot()
#Plots.scatter!(pp, (let x=rand(test, 8000); x[1,:], x[2,:] end)..., alpha=0.1, markerstrokealpha=0.0, color=:red, label="original")
#Plots.scatter!(pp, (let x=rand(fit, 8000); x[1,:], x[2,:] end)..., alpha=0.1, markerstrokealpha=0.0, color=:blue, label="fitted")
#pp


df_original, _ = get_original_and_weighted_df(test, weight_func; N=8_000)
non_weight_columns = [col for col in Symbol.(names(df_original)) if col != :weight]

pairplot(df_original[!, non_weight_columns], create_df(fit)[!, non_weight_columns])

