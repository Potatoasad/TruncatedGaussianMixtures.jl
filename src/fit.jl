function fit_gmm_2D(data, n_comps; bounds1=[0.0,1.0], bounds2=[0.0,1.0],tol=1e-6)
	EM = ExpectationMaximization(data,n_comps, bounds1=bounds1, bounds2=bounds2, tol=tol)
	MAX_REPS = 1000
	N = size(data)[2]
	old_score = Inf
	converge = abs(EM.score - old_score) ≤ tol
	for i ∈ 1:10
		update!(EM)
		converge = abs(EM.score - old_score)/N ≤ tol
		old_score = EM.score
		println(EM.score)
	end
	reps = 10;
	while (!EM.converged || old_score > 0.0) && (reps < MAX_REPS)
		update!(EM)
		reps += 1
		converge = abs(EM.score - old_score)/N ≤ tol
		old_score = EM.score
		println(EM.score)
	end

	dist(EM.mix)
end
