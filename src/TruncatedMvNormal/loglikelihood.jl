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
	if all(a .== -Inf) & all(b .== -Inf)
		return zero(a[1])
	end
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
# Specific implementation of sampling from diagonal TruncatedMvNormal
diagnormal_dists(d) = [truncated(Normal(μ, √(Σ)),a, b)  for (μ,Σ,a,b) ∈ zip(d.normal.μ,diag(d.normal.Σ),d.a, d.b)]


function _neff_and_weights(logw)
    m = maximum(logw)
    w = exp.(logw .- m)
    w ./= sum(w)
    return inv(sum(abs2, w)), w
end

function _importance_sample_truncmv!(
    rng::AbstractRNG,
    d::TruncatedMvNormal,
    x::AbstractMatrix{T};
    drawfactor::Int = 50
) where {T <: Real}

    D = length(d)
    Nout = size(x,2)

    # proposal: diagonal truncated normals
    dists = diagnormal_dists(d)

    # untruncated diagonal normals (for logq)
    μ = d.normal.μ
    Σdiag = diag(d.normal.Σ)
    diag_normals = [Normal(μ[k], sqrt(Σdiag[k])) for k in 1:D]

    maxdraws = drawfactor * Nout
    batchsize = max(4*Nout, 128)

    proposals = Matrix{T}(undef, D, maxdraws)
    logw = Vector{T}(undef, maxdraws)

    val = Vector{T}(undef, D)
    ndraws = 0

    nbatches = cld(maxdraws, batchsize)

    for _ in 1:nbatches

        nnew = min(batchsize, maxdraws - ndraws)
        nnew == 0 && break

        for _ in 1:nnew

            logq = zero(T)

            @inbounds for k in 1:D
                val[k] = Distributions.rand(rng, dists[k])
                logq += logpdf(diag_normals[k], val[k])
            end

            ndraws += 1
            proposals[:, ndraws] .= val

            logw[ndraws] = logpdf(d.normal, val) - logq
        end

        # compute normalized weights safely
        lw = view(logw,1:ndraws)
        m = maximum(lw)
        w = exp.(lw .- m)
        w ./= sum(w)

        neff = inv(sum(abs2, w))

        if neff >= Nout
            idx = Distributions.rand(rng, Categorical(w), Nout)
            @inbounds for j in 1:Nout
                x[:,j] .= proposals[:, idx[j]]
            end
            return x
        end
    end

    # fallback if ESS target not reached
    lw = view(logw,1:ndraws)
    m = maximum(lw)
    w = exp.(lw .- m)
    w ./= sum(w)

    neff = inv(sum(abs2, w))
    @warn "importance sampler hit maxdraws" neff ndraws maxdraws

    idx = Distributions.rand(rng, Categorical(w), Nout)
    @inbounds for j in 1:Nout
        x[:,j] .= proposals[:, idx[j]]
    end

    return x
end

function _importance_sample_truncmv!(
    rng::AbstractRNG,
    d::TruncatedMvNormal,
    x::AbstractVector{T};
    drawfactor::Int = 50
) where {T <: Real}

    X = reshape(x, :, 1)
    _importance_sample_truncmv!(rng, d, X; drawfactor=drawfactor)

    return x
end

"""
rand(TruncatedMvNormal(MvNormal([-2.0, -2.0], [1.0 0.1;0.1 1.0]),[0.0, 0.0], [1.0, 1.0]))
"""


function Distributions._rand!(rng::AbstractRNG, d::TruncatedMvNormal, x::AbstractArray{T}) where {T <: Real}

    if d.logtp > -3.0
        val = Vector{T}(undef,length(d))

        for i in 1:size(x,2)
            accepted = false
            while !accepted
                Distributions.rand!(rng, d.normal, val)
                accepted = Distributions.insupport(d,val)
            end
            x[:,i] .= val
        end

        return x
    else
        return _importance_sample_truncmv!(rng, d, x)
    end
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
"""

function Distributions._rand!(rng::AbstractRNG, d::TruncatedMvNormal{L}, x::AbstractArray{T}) where {T <: Real, L <: Distributions.DiagNormal}
    if d.logtp > -3.0
        for i ∈ 1:size(x,2)
            accepted = false
            val = zeros(eltype(d),length(d))
            while !accepted
                val = Distributions.rand(rng, d.normal)
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
