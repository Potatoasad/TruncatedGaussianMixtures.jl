abstract type AbstractCovarianceStructure end

struct DiagonalCovariance <: AbstractCovarianceStructure end
struct FullCovariance <: AbstractCovarianceStructure end

covariance_type(x::TruncatedMvNormal{DiagNormal}) = DiagonalCovariance
covariance_type(x::TruncatedMvNormal{FullNormal}) = FullCovariance
covariance_type(x::MixtureModel) = covariance_type(x.components[1])

mat2vecs(A) = [A[:,i] for i in 1:size(A,2)]

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

function initialize(X,K, ::FullCovariance)
	Result = kmeans(X,K)
	d,_ = size(X)
	z = Result.assignments
	μs = zeros(d,K);
	Σs = zeros(d,d, K);
	ηs = zeros(K)
	means = Result.centers |> mat2vecs
	for k ∈ 1:K
		subset = @view X[:, z .== k ]
		subset_mean_rem = subset .- mean(subset, dims=2)
		μs[:, k] = means[k][:]
		ηs[k] = length(subset[1,:])/size(X)[2]
		Σs[:,:,k] = (subset_mean_rem * subset_mean_rem') ./ length(subset[1,:])
	end
	μs,Σs,ηs
end

function initialize(X,K, ::DiagonalCovariance)
	Result = kmeans(X,K)
	d,_ = size(X)
	z = Result.assignments
	μs = zeros(d,K);
	Σs = zeros(d,K);
	ηs = zeros(K)
	means = Result.centers |> mat2vecs
	for k ∈ 1:K
		subset = @view X[:, z .== k ]
		subset_mean_rem = subset .- mean(subset, dims=2)
		μs[:,k] = means[k][:]
		ηs[k] = length(subset[1,:])/size(X)[2]
		Σs[:,k] = diag((subset_mean_rem * subset_mean_rem') ./ length(subset[1,:]))
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

function initialize(X,N, a, b; cov=:diag)
	if cov == :diag
		μs,Σs,ηs = initialize(X,N, DiagonalCovariance())
	elseif cov == :full
		μs,Σs,ηs = initialize(X,N, FullCovariance())
	end
	#w = rand(N)
	#μ1,σ1,μ2,σ2,η = (rand(N),rand(N),rand(N),rand(N),w./(sum(w)))
	#TruncatedMixture(μ1,μ2,σ1,σ2,η,bounds1,bounds2)
	make_mixture(μs, Σs, ηs, a, b)
end

DiagonalCovMixtureType = MixtureModel{A,B,TruncatedMvNormal{DiagNormal,T}} where {A,B,T <: Real}
FullCov = MixtureModel{A,B, TruncatedMvNormal{FullNormal,T}} where {A,B,T <: Real}