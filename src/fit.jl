import TruncatedGaussianMixtures
using ProgressMeter

function fit_gmm(X, K, a, b; cov=:full, tol=1e-2, MAX_REPS=100, verbose=false, progress=false, responsibilities=false, block_structure=false)
	if progress
		progressbar = Progress(MAX_REPS)
	end
	EM = ExpectationMaximization(X,K, a=a, b=b, cov=cov, block_structure=block_structure)
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


function fit_gmm(X, K, a, b, S::AbstractSchedule; cov=:full, tol=1e-2, MAX_REPS=100, verbose=false, progress=false, responsibilities=false, block_structure=false, convergence=false)
	N = length(iterator(S))
	if progress
		progressbar = Progress(N)
	end
	@show N
	EM = ExpectationMaximization(X,K, a=a, b=b, cov=cov, block_structure=block_structure)
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