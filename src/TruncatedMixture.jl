
struct TruncatedMixture{T}
	μ1::Vector{T}
	μ2::Vector{T}
	σ1::Vector{T}
	σ2::Vector{T}
	η::Vector{T}
	bounds1::Vector{T}
	bounds2::Vector{T}
end

deepcopy2(x::T) where T = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)

function Zⁿ(x::TruncatedMixture, datapoint::Vector)
	N_kernels = length(x.μ1)
	unnormalized = zeros(N_kernels);
	for k ∈ 1:N_kernels
		kernelR = truncated(Normal(x.μ1[k], x.σ1[k]), x.bounds1[1], x.bounds1[2]);
		kernelΦ = truncated(Normal(x.μ2[k], x.σ2[k]), x.bounds2[1], x.bounds2[2]);
		Y = datapoint;
		unnormalized[k] = log(x.η[k]) + loglikelihood(kernelR, Y[1]) + loglikelihood(kernelΦ, Y[2])
	end
	return exp.(unnormalized .- LogExpFunctions.logsumexp(unnormalized))
	#return exp.(unnormalized)./sum(exp.(unnormalized))
end

TruncatedMixture(R::Vector{N}, ϕ::Vector{M}, η::Vector, bounds1, bounds2) where {N <: Distribution, M <: Distribution} = TruncatedMixture(
	[d.μ for d ∈ R],
	[d.μ for d ∈ ϕ],
	[d.σ for d ∈ R],
	[d.σ for d ∈ ϕ],
	η,
	Float64.(bounds1),
	Float64.(bounds2)
)
	
	
TruncatedMixture(R::Vector{N}, ϕ::Vector{M}, η::Vector) where {N <: Truncated, M <: Truncated} = TruncatedMixture(
	[d.untruncated.μ for d ∈ R],
	[d.untruncated.μ for d ∈ ϕ],
	[d.untruncated.σ for d ∈ R],
	[d.untruncated.σ for d ∈ ϕ],
	η,
	[R[1].lower, R[1].upper],
	[ϕ[1].lower, ϕ[1].upper]
)


function kernel(μ1::T,σ1::T,bounds1::Vector{T}) where {T <: Real}
	truncated(Normal(μ1,σ1),bounds1[1],bounds1[2])
end

function kernel(μ1::T,σ1::T,bounds1::Vector{T},μ2::T,σ2::T,bounds2::Vector{T}) where {T <: Real}
	Kern1 = truncated(Normal(μ1,σ1),bounds1[1],bounds1[2])
	Kern2 = truncated(Normal(μ2,σ2),bounds2[1],bounds2[2])
	product_distribution([Kern1,Kern2])
end
	
dist(x::TruncatedMixture) = MixtureModel([kernel(x.μ1[i],x.σ1[i],x.bounds1,x.μ2[i],x.σ2[i],x.bounds2) for i=1:length(x.μ1)], x.η)