struct TruncatedMvNormal{K <: MvNormal, T <: Real} <: ContinuousMultivariateDistribution
	normal::K
	a::Vector{T}
	b::Vector{T}
	logtp::Float64 # Total probability inside bounds
end

dim_of_normal(x::MvNormal) = length(x.μ)

function log_mvnormcdf(dist::DiagNormal, a, b)
	total = 0.0
	for i ∈ 1:dim_of_normal(dist)
		dnorm = Normal(dist.μ[i], dist.Σ[i,i] > zero(dist.Σ[i,i]) ? sqrt(dist.Σ[i,i]) : 10*eps(dist.Σ[i,i]))
		total += logsubexp(logcdf(dnorm, b[i]),logcdf(dnorm, a[i]))
	end
	total
end

function log_mvnormcdf(dist::FullNormal, a, b)
	x,_ = mvnormcdf(dist, a, b)
	log(x)
end

log_mvnormcdf(x::TruncatedMvNormal) = x.logtp

TruncatedMvNormal(normal::MvNormal, a::Vector, b::Vector) = TruncatedMvNormal(normal, a, b, log_mvnormcdf(normal, a, b))


function a_and_b_from_bounds(bounds)
	a = zero(bounds[1]); b = zero(bounds[1]);
	for i ∈ 1:length(bounds)
		a[i] = bounds[i][1]
		b[i] = bounds[i][2]
	end
	a,b
end

TruncatedMvNormal(x::T, bounds::Vector{Vector{L}}) where {T <: Distributions.AbstractMvNormal, L <: Real} = TruncatedMvNormal(x,a_and_b_from_bounds(bounds)...)

Base.length(d::TruncatedMvNormal) = Base.length(d.normal.μ)

Distributions.sampler(d::TruncatedMvNormal) = d
	
Base.eltype(d::TruncatedMvNormal) = Base.eltype(d.normal)

function Distributions.insupport(d::TruncatedMvNormal, x::AbstractVector)
	inside = true
	for i ∈ 1:length(d)
		inside = inside && (d.a[i] ≤ x[i] ≤ d.b[i])
	end
	return inside
end

function Distributions.insupport(d::TruncatedMvNormal, x::AbstractMatrix)
	inside = zeros(Bool, size(x,2))
	for i ∈ 1:size(x,2)
		inside[i] = Distributions.insupport(d, x[:,i])
	end
	return inside
end

function Distributions.insupport(d::TruncatedMvNormal, x::AbstractArray)
	inside = zeros(Bool, size(x,2))
	for i ∈ 1:size(x,2)
		inside[i] = Distributions.insupport(d, x[:,i])
	end
	return inside
end

"""
# Naive implementation of sampling from normal
function Distributions._rand!(rng::AbstractRNG, d::TruncatedMvNormal, x::AbstractArray{T}) where {T <: Real}
	for i ∈ 1:size(x,2)
		accepted = false
		val = zeros(eltype(d),length(d))
		while !accepted
			val = rand(rng, d.normal)
			accepted = Distributions.insupport(d, val)
		end
		x[:,i] = val
	end
	x
end
"""


# Fast but innacurate implementation of sampling from normal coupled with some rejection sampling
function Distributions._rand!(rng::AbstractRNG, d::TruncatedMvNormal, x::AbstractArray{T}) where {T <: Real}
    if d.logtp > -3.0
        for i ∈ 1:size(x,2)
            accepted = false
            val = zeros(eltype(d),length(d))
            while !accepted
                val = rand(rng, d.normal)
                accepted = Distributions.insupport(d, val)
            end
            x[:,i] = val
        end
    else
        dists = diagnormal_dists(d)
		for i ∈ 1:size(x,2)
			accepted = false
			val = zeros(eltype(d),length(d))
			c = 1.4;
			while !accepted
				diaglogpdf = zero(eltype(d))
				for k ∈ 1:length(d)
		           Distributions._rand!(rng, dists[k], @view(val[k]))
					diaglogpdf += logpdf(dists[k], val[k])
		        end
				fulllogpdf = logpdf(d, val)
				ratio = exp(fulllogpdf - diaglogpdf)
				u = rand()
				if u < ratio/c
					accepted = true
					c = max(c, ratio)
				end
			end
			x[:,i] = val
		end
    end
    x
end

# Specific implementation of sampling from diagonal TruncatedMvNormal
diagnormal_dists(d) = [truncated(Normal(μ, √(Σ)),a, b)  for (μ,Σ,a,b) ∈ zip(d.normal.μ,diag(d.normal.Σ),d.a, d.b)]

function Distributions._rand!(rng::AbstractRNG, d::TruncatedMvNormal{L}, x::AbstractArray{T}) where {T <: Real, L <: Distributions.DiagNormal}
    if d.logtp > -3.0
        for i ∈ 1:size(x,2)
            accepted = false
            val = zeros(eltype(d),length(d))
            while !accepted
                val = rand(rng, d.normal)
                accepted = Distributions.insupport(d, val)
            end
            x[:,i] = val
        end
    else
        dists = diagnormal_dists(d)
        for k ∈ 1:length(d)
           k_dim_slice = @view x[k,:]
           Distributions._rand!(rng, dists[k], k_dim_slice)
        end
    end
    x
end

function Distributions._logpdf(d::TruncatedMvNormal, x::AbstractVector)
	if !Distributions.insupport(d,x)
		return typemin(eltype(d))
	end
	Distributions._logpdf(d.normal,x) .- log_mvnormcdf(d)
end

function Distributions._logpdf(d::TruncatedMvNormal, x::AbstractMatrix)
	p = zeros(size(x,2))
	for i ∈ 1:size(x,2)
		p[i] = Distributions._logpdf(d, x[:,i])
	end
	p
end

function Distributions._logpdf(d::TruncatedMvNormal, x::AbstractArray)
	p = zeros(size(x)[2:end])
	for z ∈ CartesianIndices(size(p))
		p[z] = Distributions._logpdf(d, x[:,z])
	end
	p
end
