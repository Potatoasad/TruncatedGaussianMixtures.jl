function initialize(X,N)
	Result = kmeans(X,N)
	z = Result.assignments
	μ1 = zeros(N); μ2 = zeros(N);
	σ1 = zeros(N); σ2 = zeros(N);
	η = zeros(N)
	means = Result.centers |> mat2vecs
	for k ∈ 1:N
		subset_1 = @view X[1, z .== k ]
		subset_2 = @view X[2, z .== k ]
		μ1[k] = means[k][1]
		σ1[k] = std(subset_1)
		μ2[k] = means[k][2]
		σ2[k] = std(subset_2)
		η[k] = length(subset_1)/size(X)[2]
	end
	μ1,σ1,μ2,σ2,η
end

function initialize(X,N, bounds1, bounds2)
	μ1,σ1,μ2,σ2,η = initialize(X,N)
	#w = rand(N)
	#μ1,σ1,μ2,σ2,η = (rand(N),rand(N),rand(N),rand(N),w./(sum(w)))
	TruncatedMixture(μ1,μ2,σ1,σ2,η,bounds1,bounds2)
end