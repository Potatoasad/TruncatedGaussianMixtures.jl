abstract type AbstractDataset end

abstract type AbstractUnbiasingType end

struct DefaultUnbiasing <: AbstractUnbiasingType end

struct BoundaryUnbiasing{T <: AbstractVector, L, K} <: AbstractUnbiasingType 
	columns::T
	bandwidth_scale::K
	bandwidth_dimension::L
end

BoundaryUnbiasing(columns) = BoundaryUnbiasing(columns, length(columns), 1.0)
BoundaryUnbiasing(columns, bandwidth_scale::Real) = BoundaryUnbiasing(columns, bandwidth_scale, length(columns))
BoundaryUnbiasing(columns, bandwidth_scale::Real, bandwidth_dimension::Real) = BoundaryUnbiasing(columns, bandwidth_scale, bandwidth_dimension)

import Base


function get_data(x::AbstractVector)
	return x
end

function Base.length(x::AbstractDataset)
	return Base.length(get_data(x))
end

function Base.iterate(x::AbstractDataset)
	Base.iterate(get_data(x))
end

function Base.iterate(x::AbstractDataset, state)
	Base.iterate(get_data(x), state)
end


function Base.size(x::AbstractDataset)
    return size(get_data(x))
end

function Base.size(x::AbstractDataset, args...)
    return size(get_data(x), args...)
end

function Base.getindex(x::AbstractDataset, args...)
    return Base.getindex(get_data(x), args...)
end

Base.isdone(iter::AbstractDataset) = Base.isdone(iter)
Base.isdone(iter::AbstractDataset, state) = Base.isdone(iter, state)


struct StandardData{T} <: AbstractDataset
	yₙ::T
	N_data::Int64
end

function Base.eltype(::Type{StandardData{T}}) where {T}
    return Base.eltype(T)
end

function get_data(x::StandardData)
	return x.yₙ
end

function StandardData(yₙ)
	StandardData(yₙ, length(yₙ))
end

fixtype(x::StandardData) = StandardData(fixtype(get_data(x)))

function mean_and_cov(data::StandardData, zⁿₖ, W, k::Int)
	normalization = sum(zⁿₖ[n][k]* W[n] for n ∈ 1:data.N_data)
	themean = sum(zⁿₖ[n][k]*data.yₙ[n]* W[n] for n ∈ 1:data.N_data)/(normalization)
	thecov = sum(zⁿₖ[n][k]*W[n]*(data.yₙ[n]-themean)*(data.yₙ[n]-themean)' for n ∈ 1:data.N_data)/(normalization)
	themean, thecov
end


struct BoundaryUnbiasedData{T, L <: AbstractMatrix, K <: AbstractArray, P <: AbstractVector} <: AbstractDataset
	yₙ::T
	N_data::Int64
	bandwidth::P
	μ::L
	M1::L
	M2::K
end

fixtype(x::BoundaryUnbiasedData) = BoundaryUnbiasedData(fixtype(get_data(x)), length(fixtype(get_data(x))), x.bandwidth, x.μ, x.M1, x.M2)

#function BoundaryUnbiasedData(yₙ::T, N_data, bandwidth::P, μ::L, M1::L, M2::K)
#	BoundaryUnbiasedData{eltype(yₙ), }(yₙ, N_data, bandwidth, μ, M1, M2)
#end

function Base.eltype(::Type{BoundaryUnbiasedData{T, L, K, P}}) where {T, L, K, P}
    return Base.eltype(T)
end

function get_data(x::BoundaryUnbiasedData)
	return x.yₙ
end

ϕ(x) = pdf(NORMALDIST,x)
Φ(x) = cdf(NORMALDIST,x)

function xa(μ, σ₀, a, b)
    α = (a - μ)/σ₀; β = (b - μ)/σ₀;
    return (b - σ₀*(β*Φ(β) + ϕ(β)) + σ₀*(α*Φ(α) + ϕ(α)))
end

function get_best_position(x,σ₀,a,b)
    find_zero(μ -> (xa(μ,σ₀,a,b)-x), x)
end

function compute_bandwidth(yₙ, dim)
	d = length(yₙ[1])
    N = length(yₙ)
    stds = [std(yₙ[n][i] for n ∈ 1:N) for i ∈ 1:d]
    d = dim;
    (4/(d+2))^(1/(d+4)).*stds.*N.^(-1/(d+4)) # silvermans rule
end

function compute_stds(yₙ, dim)
	d = length(yₙ[1])
    N = length(yₙ)
    stds = [std(yₙ[n][i] for n ∈ 1:N) for i ∈ 1:d]
    stds
end

function compute_bandwidth(yₙ)
    d = length(yₙ[1])
    compute_bandwidth(yₙ, d)
end

function mean_and_var_1d(mu, sig, a, b)
    dist = truncated(Normal(mu, sig), a, b)
    mean(dist), var(dist)
end

function BoundaryUnbiasedData(yₙ, a, b, unbiasing::BoundaryUnbiasing; bandwidth_scale=nothing)
	N = length(yₙ);
	d = length(yₙ[1])
	T = eltype(yₙ[1])


	μₙ = zeros(T, N, d)
	M1 = zeros(T, N, d)
	M2 = zeros(T, N, d, d)

	if bandwidth_scale == nothing
		σ₀ = compute_bandwidth(yₙ, unbiasing.bandwidth_dimension)
	elseif unbiasing.bandwidth_scale != nothing
		σ₀ = bandwidth_scale .* compute_bandwidth(yₙ, unbiasing.bandwidth_dimension) # Scale relative to silverman's rule
	else
		σ₀ = bandwidth_scale .* compute_stds(yₙ, d) # Scale relative to std devs
	end

	for n in 1:N
		for i in 1:d
			if unbiasing.columns[i] == 1
				μₙ[n,i] = get_best_position(yₙ[n][i], σ₀[i], a[i], b[i])
			else
				μₙ[n,i] = yₙ[n][i]
			end
			m,v = mean_and_var_1d(μₙ[n,i], σ₀[i], a[i], b[i])
			M1[n,i] = m
			M2[n,i,i] = v + m^2 
		end

		for i in 1:d, j in 1:d
			if i !==j
				M2[n,i,j] = M1[n,i] * M1[n,j] 
			end
		end
	end

	BoundaryUnbiasedData(yₙ, N, σ₀, μₙ, M1, M2)
end


function mean_and_cov(data::BoundaryUnbiasedData, zⁿₖ::AbstractVector, W, k::Int)
	normalization = sum(zⁿₖ[n][k]* W[n] for n ∈ 1:data.N_data)

	themean = sum(zⁿₖ[n][k]*@view(data.M1[n, :])* W[n] for n ∈ 1:data.N_data)/(normalization)
	theM2 = sum(zⁿₖ[n][k]*W[n]*@view(data.M2[n, :, :]) for n ∈ 1:data.N_data)/(normalization)
	thecov = theM2 - themean * themean'
	themean, thecov
end

function mean_and_cov(data::BoundaryUnbiasedData, zⁿₖ::AbstractMatrix, W, k::Int)
	normalization = sum(zⁿₖ[n,k]* W[n] for n ∈ 1:data.N_data)

	themean = sum(zⁿₖ[n,k]*@view(data.M1[n, :])* W[n] for n ∈ 1:data.N_data)/(normalization)
	theM2 = sum(zⁿₖ[n,k]*W[n]*@view(data.M2[n, :, :]) for n ∈ 1:data.N_data)/(normalization)
	thecov = theM2 - themean * themean'
	themean, thecov
end