using Distributions, LinearAlgebra, TruncatedGaussianMixtures

a = [0.0, 0.0]; b = [1.0, 1.0]
μ = [0.5, 0.0]; Σ = Diagonal([0.2, 0.8])
dist = TruncatedMvNormal(MvNormal(μ, Σ), a, b)
X = rand(dist, 8000)

EM = TruncatedGaussianMixtures.fit_gmm(X, 10, a, b, AnnealingSchedule(β_max=1.0, dβ_rise=1.0, N_post=2000); cov=:diag, convergence=true, tol=1e-7, responsibilities=true, progress=true)

using Makie, GLMakie, PairPlots, DataFrames

make_df(x::Distributions.Distribution) = DataFrame(collect(rand(x, 8000)'), :auto)
make_df(x::DataFrames.DataFrame) = x

function compare_distributions(dist1, dist2; columns=:auto)
	X1 = make_df(dist1)
	X2 = DataFrame(collect(rand(dist2, 8000)'), names(X1))

	if columns isa Symbol
		columns = Dict([Symbol(s) => s for s in names(X1)])
	else
		columns = Dict([Symbol(s) => c for (s,c) in zip(names(X1), columns)])
	end

	@show columns


	fig = Figure()
	gs = GridLayout(fig[1,1])

	c1, c2 = Makie.wong_colors(0.5)[1:2]
	plot_attributes(color) = (
        PairPlots.Scatter(color=color),
        PairPlots.Contourf(color=color),

        # New:
        PairPlots.MarginHist(color=color),
        PairPlots.MarginConfidenceLimits(color=color),
    )

	pairplot(gs, PairPlots.Series(X1, label="Truth", color=c1, strokecolor=c1) => plot_attributes(c1),
			 	 PairPlots.Series(X2, label="Fit"  , color=c2, strokecolor=c2) => plot_attributes(c2),
			 	 labels = columns)


	#rowgap!(gs, 10)
	#colgap!(gs, 10)

	fig
end

compare_distributions(dist, EM.mix, columns=[L"\chi_1", L"\chi_2"])


