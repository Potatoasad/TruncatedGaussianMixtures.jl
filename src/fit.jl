function fit_gmm_2D(data, n_comps; bounds1=[0.0,1.0], bounds2=[0.0,1.0],tol=1e-2, MAX_REPS=100, verbose=false)
	EM = ExpectationMaximization(data,n_comps, bounds1=bounds1, bounds2=bounds2, tol=tol)
	N = size(data)[2]
	old_score = Inf
	converge = abs(EM.score - old_score) ≤ tol
	for i ∈ 1:10
		update!(EM)
		converge = abs(EM.score - old_score) ≤ tol
		old_score = EM.score
		if verbose
			println(EM.score)
		end
	end
	reps = 10;
	while (!EM.converged || old_score > 0.0) && (reps < MAX_REPS)
		update!(EM)
		reps += 1
		converge = abs(EM.score - old_score) ≤ tol
		old_score = EM.score
		if verbose
			println(EM.score)
		end
	end

	dist(EM.mix)
end
