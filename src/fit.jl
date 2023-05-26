function fit_gmm(X, K, a, b; cov=:full, tol=1e-2, MAX_REPS=100, verbose=false, progress=false)
	EM = ExpectationMaximization(X,K, a=a, b=b, cov=cov)
	old_score = Inf
	converge = abs(EM.score - old_score) ≤ tol
	if progress
		progressbar = Progress(MAX_REPS)
	end
	for i ∈ 1:10
		update!(EM)
		converge = (abs(EM.score - old_score)/abs(old_score)) ≤ tol
		if verbose
			println("Score: ", EM.score, "   |ΔScore|/|Score| = ", abs(EM.score - old_score)/abs(old_score))
		end
		old_score = EM.score
		if progress
			next!(progressbar)
		end
	end
	reps = 10;
	while (!converge || old_score > 0.0) && (reps < MAX_REPS)
		update!(EM)
		reps += 1
		converge = (abs(EM.score - old_score)/abs(old_score)) ≤ tol
		if verbose
			println("Score: ", EM.score, "   |ΔScore|/|Score| = ", abs(EM.score - old_score)/abs(old_score))
		end
		old_score = EM.score
		if progress
			next!(progressbar)
		end
	end
	if progress
		finish!(progressbar)
	end
	EM.mix
end