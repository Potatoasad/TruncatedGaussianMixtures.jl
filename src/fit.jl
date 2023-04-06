function fit_gmm(X, K, a, b; cov=:full, tol=1e-2, MAX_REPS=100, verbose=false)
	EM = ExpectationMaximization(X,K, a=a, b=b, cov=cov)
	old_score = Inf
	converge = abs(EM.score - old_score) ≤ tol
	for i ∈ 1:10
		update!(EM)
		converge = abs(EM.score - old_score) ≤ tol
		if verbose
			println(abs(EM.score - old_score))
		end
		old_score = EM.score
	end
	reps = 10;
	while (!converge || old_score > 0.0) && (reps < MAX_REPS)
		update!(EM)
		reps += 1
		converge = (abs(EM.score - old_score)/old_score) ≤ tol
		if verbose
			println(abs(EM.score - old_score))
		end
		old_score = EM.score
	end
	EM.mix
end