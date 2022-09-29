function fit_gmm_2D(data, n_comps; bounds1=[0.0,1.0], bounds2=[0.0,1.0],tol=1e-16)

	EM = ExpectationMaximization(data,n_comps)
	MAX_REPS = 1000
	for i ∈ 1:10
		update!(EM)
	end
	reps = 10;
	while !EM.converged && (reps < MAX_REPS)
		update!(EM)
		reps += 1
	end

	dist(EM.mix)
end