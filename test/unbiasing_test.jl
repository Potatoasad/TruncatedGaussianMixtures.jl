using Plots, Distributions, LinearAlgebra, Test
using Revise, TruncatedGaussianMixtures
using PairPlots
using DataFrames
using CairoMakie
using ProgressMeter

df = DataFrame(2 .* π .* rand(8000,2), :auto)
df[!, :x1] = let x=df[!, :x1]; @. x end
df[!, :x2] = let x=df[!, :x2]; @. x end

a = zeros(2); b = 2 .* π .* ones(2);
EM_bound = fit_gmm(df, 6, a, b; 
					cov=:full, tol=1e-8, 
					MAX_REPS=100, progress=true, 
					responsibilities=true,
					unbiasing=BoundaryUnbiasing([1,1]));

EM = fit_gmm(df, 6, a, b; cov=:full, tol=1e-2, MAX_REPS=10, progress=true, responsibilities=true);

EM_full = ExpectationMaximization(
	EM.data,
	deepcopy(EM_bound.mix),
	EM_bound.tol,
	deepcopy(EM_bound.zⁿₖ),
	deepcopy(EM_bound.score),
	EM_bound.converged,
	EM_bound.cov_type,
	EM_bound.block_structure,
	EM_bound.weights
)

@showprogress for i in 1:100
	TruncatedGaussianMixtures.update!(EM_full)
end


fit = EM.mix

pp = Plots.plot()
Plots.scatter!(pp, (let x=df; df[!, :x1], df[!, :x2] end)..., alpha=0.1, markerstrokealpha=0.0, color=:red, label="original")
Plots.scatter!(pp, (let x=rand(fit, 8000); x[1,:], x[2,:] end)..., alpha=0.1, markerstrokealpha=0.0, color=:blue, label="fitted")
savefig("test.png")

