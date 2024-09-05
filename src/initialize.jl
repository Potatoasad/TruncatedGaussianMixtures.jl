abstract type AbstractCovarianceStructure end

struct DiagonalCovariance <: AbstractCovarianceStructure end
struct FullCovariance <: AbstractCovarianceStructure end

covariance_type(x::TruncatedMvNormal{DiagNormal}) = DiagonalCovariance
covariance_type(x::TruncatedMvNormal{FullNormal}) = FullCovariance
covariance_type(x::MixtureModel) = covariance_type(x.components[1])

mat2vecs(A) = [A[:,i] for i in 1:size(A,2)]
vec2mat(X) = X
vec2mat(X::AbstractVector) = hcat(X...)
vec2mat(X::AbstractDataset)= vec2mat(get_data(X))
weight_vector(N) = StatsBase.Weights(ones(N)./N)

function initialize_like(AB::MixtureModel{A,B,TruncatedMvNormal{DiagNormal,T}}) where {A,B,T <: Real}
	d = length(AB); K = n_components(AB)
	Σsshape = (d,K)
	μsshape = (d,K)
	μs = zeros(μsshape); Σs = zeros(Σsshape); ηs = zeros(K)
	#get_params!(μs, Σs, ηs, AB)
	μs, Σs, ηs
end

function initialize_like(AB::MixtureModel{A,B,TruncatedMvNormal{FullNormal,T}}) where {A,B,T <: Real}
	d = length(AB); K = n_components(AB)
	Σsshape = (d,d,K)
	μsshape = (d,K)
	μs = zeros(μsshape); Σs = zeros(Σsshape); ηs = zeros(K)
	#get_params!(μs, Σs, ηs, AB)
	μs, Σs, ηs
end

function fix_zero_covariances(Σ::AbstractMatrix, a, b, N)
	if det(Σ) <= 0.0
		Σ = Σ .+  diagm((b-a)./N).^2
	else
		return Σ
	end
end

function fix_zero_covariances(Σ::AbstractVector, a, b, N)
	if any(Σ .<= zero(eltype(Σ)))
		Σ = Σ .+  ((b-a)./N).^2
	else
		return Σ
	end
end


compute_weights(weights::Nothing, N) = weight_vector(N)
compute_weights(weights, N) = StatsBase.Weights(weights ./ weights.sum)

function get_subset_mean_remainder(X, W)
	D,N = size(X)
	X_mean = ones(D)
	W_sum = sum(W)
	for d ∈ 1:D
		X_mean[d] = sum( X[d,:] .* W ) / W_sum
	end

	X_remainder = ones(size(X))
	N = size(X,2)
	for i ∈ 1:N
		X_remainder[:,i] .= (X[:,i] .- X_mean)
	end
	X_remainder
end

function get_subset_covariances(rem,W)
	D,N = size(rem)
	cov = zeros(D,D)
	for n ∈ 1:N
		for i ∈ 1:D, j ∈ 1:D
			cov[i,j] += rem[i,n] * rem[j,n] * W[n]
		end
	end
	return cov ./ sum(W)
end

function initialize(X_in,K,::FullCovariance; weights=nothing)
	X = vec2mat(X_in)
	N = size(X,2);
	weights = compute_weights(weights, N)

	Result = kmeans(X,K; weights=weights.values)
	a,b = minimum(X, dims=2), maximum(X, dims=2);
	a = reshape(a, length(a))
	b = reshape(b, length(b))
	d,_ = size(X)
	z = Result.assignments
	μs = zeros(d,K);
	Σs = zeros(d,d,K);
	ηs = zeros(K)
	means = Result.centers |> mat2vecs
	for k ∈ 1:K
		subset = @view X[:, z .== k ]  # dxN_k
		subset_weights = @view weights[z .== k]  # N_k
		#subset_mean_rem =e subset .- mean(subset, dims=2)
		subset_mean_rem = get_subset_mean_remainder(subset, subset_weights)
		μs[:, k] = means[k][:]
		ηs[k] = length(subset[1,:])/size(X)[2]
		#Σs[:,:,k] = fix_zero_covariances((subset_mean_rem * subset_mean_rem') ./ length(subset[1,:]),a,b,N)
		#Σs[:,:,k] = fix_zero_covariances(get_subset_covariances(subset_mean_rem,subset_weights),a,b,N)
		Σs[:,:,k] = get_subset_covariances(subset_mean_rem,subset_weights)
		Σs[:,:,k] = @. (Σs[:,:,k] + Σs[:,:,k]')/2
		#println(Σs[:,:,k])
	end
	μs,Σs,ηs
end

function initialize(X_in,K, ::DiagonalCovariance; weights=nothing)
	X = vec2mat(X_in)
	#@show size(X)
	N = size(X,2);
	weights = compute_weights(weights, N)
	Result = kmeans(X,K; weights=weights.values)
	a,b = minimum(X, dims=2), maximum(X, dims=2);
	a = reshape(a, length(a))
	b = reshape(b, length(b))
	d,_ = size(X)
	z = Result.assignments
	μs = zeros(d,K);
	Σs = zeros(d,K);
	ηs = zeros(K)
	means = Result.centers |> mat2vecs
	for k ∈ 1:K
		subset = @view X[:, z .== k ]  # dxN_k
		subset_weights = @view weights[z .== k]  # N_k
		#@show var(subset[1, :])
		#@show subset_weights
		#subset = @view X[:, z .== k ]
		subset_mean_rem = get_subset_mean_remainder(subset, subset_weights)
		#@show (subset_mean_rem * diagm(subset_weights) * subset_mean_rem') ./ (sum(subset_weights))
		#subset_mean_rem = subset .- mean(subset, dims=2)
		μs[:,k] = means[k][:]
		ηs[k] = length(subset[1,:])/size(X)[2]
		#Σs[:,k] = fix_zero_covariances(diag(get_subset_covariances(subset_mean_rem,subset_weights)),a,b,N)
		#@show get_subset_covariances(subset_mean_rem,subset_weights)
		Σs[:,k] = diag(get_subset_covariances(subset_mean_rem,subset_weights))
	end
	μs,Σs,ηs
end

function make_mixture(μs::Array{T,2}, Σs::Array{T,3}, ηs::Array{T,1}, a::Array{T,1}, b::Array{T,1}) where {T <: Real}
	dists = [TruncatedMvNormal(MvNormal(μs[:,k], Σs[:,:,k]), a, b) for k ∈ 1:size(μs)[2]]
	MixtureModel(dists, ηs./sum(ηs))
end

function make_mixture(μs::Array{T,2}, Σs::Array{T,2}, ηs::Array{T,1}, a::Array{T,1}, b::Array{T,1}) where {T <: Real}
	dists = [TruncatedMvNormal(MvNormal(μs[:,k], sqrt.(Σs[:,k])), a, b) for k ∈ 1:size(μs)[2]]
	MixtureModel(dists, ηs./sum(ηs))
end

function initialize(X,N, a, b; cov=:diag, weights=nothing)
	#weights = compute_weights(weights, size(X,2))
	if cov == :diag
		μs,Σs,ηs = initialize(X,N, DiagonalCovariance(), weights=weights)
	elseif cov == :full
		μs,Σs,ηs = initialize(X,N, FullCovariance(), weights=weights)
	end
	#w = rand(N)
	#μ1,σ1,μ2,σ2,η = (rand(N),rand(N),rand(N),rand(N),w./(sum(w)))
	#TruncatedMixture(μ1,μ2,σ1,σ2,η,bounds1,bounds2)
	#println(cholesky(Σs[:,:,2]))
	#println(cholesky(Σs[:,:,1]))
	make_mixture(μs, Σs, ηs, a, b)
end

DiagonalCovMixtureType = MixtureModel{A,B,TruncatedMvNormal{DiagNormal,T}} where {A,B,T <: Real}
FullCov = MixtureModel{A,B, TruncatedMvNormal{FullNormal,T}} where {A,B,T <: Real}
