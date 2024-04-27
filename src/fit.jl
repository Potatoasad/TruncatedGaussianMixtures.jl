import TruncatedGaussianMixtures
using ProgressMeter
using DataFrames

str_to_symbol(x) = x
str_to_symbol(x::AbstractString) = Symbol(x)

function fit_gmm(X, K, a, b; cov=:full, tol=1e-2, MAX_REPS=100, verbose=false, progress=false, responsibilities=false, block_structure=false, weights=nothing)
	cov = str_to_symbol(cov)
	if progress
		progressbar = Progress(MAX_REPS)
	end
	EM = ExpectationMaximization(X,K, a=a, b=b, cov=cov, block_structure=block_structure, weights=weights)
	old_score = Inf
	converge = false
	for i ∈ 1:10
		update!(EM)
		converge = (abs(EM.score - old_score)/abs(old_score)) ≤ tol
		if verbose
			println("Score: ", EM.score, "   |ΔScore|/|Score| = ", abs(EM.score - old_score)/abs(old_score), "  converged = ", converge)
		end
		old_score = EM.score
		if progress
			next!(progressbar)
		end
	end
	reps = 10;
	while (!converge) && (reps < MAX_REPS)
		update!(EM)
		reps += 1
		converge = (abs(EM.score - old_score)/abs(old_score)) ≤ tol
		if verbose
			println("Score: ", EM.score, "   |ΔScore|/|Score| = ", abs(EM.score - old_score)/abs(old_score), "  converged = ", converge)
		end
		old_score = EM.score
		if progress
			next!(progressbar)
		end
	end
	if progress
		finish!(progressbar)
	end
	if responsibilities
		return EM
	end
	EM.mix
end


function fit_gmm(X, K, a, b, S::AbstractSchedule; cov=:full, tol=1e-2, MAX_REPS=100, verbose=false, progress=false, responsibilities=false, block_structure=false, convergence=false, weights=nothing)
	cov = str_to_symbol(cov)
	N = length(iterator(S))
	if progress
		progressbar = Progress(N)
	end
	#@show N
	EM = ExpectationMaximization(X,K, a=a, b=b, cov=cov, block_structure=block_structure, weights=weights)
	old_score = Inf
	converge = false
	for β ∈ S
		TruncatedGaussianMixtures.update!(EM, β)
		converge = (abs(EM.score - old_score)/abs(old_score)) ≤ tol
		if convergence
			if converge
				break
			end
		end
		if verbose
			println("Score: ", EM.score, "   |ΔScore|/|Score| = ", abs(EM.score - old_score)/abs(old_score), "  converged = ", converge)
		end
		old_score = EM.score
		if progress
			next!(progressbar)
		end
	end
	if progress
		finish!(progressbar)
	end
	if responsibilities
		return EM
	end
	EM.mix
end




function sample(df::AbstractDataFrame, N::Int)
	df[rand(1:nrow(df),N),:]
end

function sample(df::AbstractDataFrame, N::Int, columns)
	df[rand(1:nrow(df),N), columns]
end


function fit_gmm(df::DataFrame, K, a, b; kwargs...)
	fit_gmm(collect(Matrix(df)'), K, a, b; kwargs...)
end

function fit_gmm(df::DataFrame, K, a, b, S::AbstractSchedule; kwargs...)
	fit_gmm(collect(Matrix(df)'), K, a, b, S; kwargs...)
end

function fit_gmm(df::DataFrame, K, a, b, Tr::AbstractTransformation; kwargs...)
	df2 = forward(Tr, df)
	EM = fit_gmm(collect(Matrix(df2[!, image_columns(Tr)])'), K, a, b; kwargs..., responsibilities=true)
	df_out = DataFrame(collect(hcat(EM.data...)'), image_columns(Tr))

	## Make categorial assignment to different components of TGMM and output the dataframe with those assignments. 
	## We can then later use those to create groups
	assignments = [rand(Categorical(p)) for p ∈ EM.zⁿₖ]
	df_out[!, :components] = assignments
	for col ∈ Tr.ignore_columns
		df_out[!, col] = df[!, col]
	end
	#if (:components ∉ Tr.ignore_columns)
	#	push!(Tr.ignore_columns, :components)
	#end
	df_out = inverse(Tr, df_out)
	df_out[!, :components] = assignments
	EM.mix, df_out
end



function fit_gmm(df::DataFrame, K, a, b, Tr::AbstractTransformation, S::AbstractSchedule; kwargs...)
	df2 = forward(Tr, df)
	EM = fit_gmm(collect(Matrix(df2[!, image_columns(Tr)])'), K, a, b, S; kwargs..., responsibilities=true)
	df_out = DataFrame(collect(hcat(EM.data...)'), image_columns(Tr))

	## Make categorial assignment to different components of TGMM and output the dataframe with those assignments. 
	## We can then later use those to create groups
	assignments = [rand(Categorical(p)) for p ∈ EM.zⁿₖ]
	df_out[!, :components] = assignments
	for col ∈ Tr.ignore_columns
		df_out[!, col] = df[!, col]
	end
	#if (:components ∉ Tr.ignore_columns)
	#	push!(Tr.ignore_columns, :components)
	#end
	df_out = inverse(Tr, df_out)
	df_out[!, :components] = assignments
	EM.mix, df_out
end











