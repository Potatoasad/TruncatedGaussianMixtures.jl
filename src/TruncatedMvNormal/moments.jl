
using Statistics: mean, cov

function μ₋ₖₖ(k::Int, x::Real, Σ::AbstractMatrix)
    Σ[Not(k),k] * inv(Σ[k,k]) * x
end

function μ₋ₖₖ(kq::AbstractVector{Int}, x::AbstractVector{T}, Σ::AbstractMatrix) where {T <: Real}
    Σ[Not(kq),kq] * inv(Σ[kq,kq]) * x
end
    
function Σ₋ₖₖ(k::Int, Σ::AbstractMatrix)
    Σ[Not(k),Not(k)] .- (Σ[Not(k),k] * inv(Σ[k,k]) * Σ[k, Not(k)]')
end

function Σ₋ₖₖ(kq::AbstractVector{Int}, Σ::AbstractMatrix)
    Σ[Not(kq),Not(kq)] .- (Σ[Not(kq),kq] * inv(Σ[kq,kq]) * Σ[kq, Not(kq)])
end

#x(x) = exp(logsumexp([log(x),log(10*eps(x))])*(1/2));
#√(x) = √(x);

function F(k::Int, x::Real, Σ::AbstractMatrix, a::AbstractVector{T}, b::AbstractVector{L}) where {T <: Real, L <: Real}
    Σ₋ₖₖ_ = Σ₋ₖₖ(k, Σ)
    μ₋ₖₖ_ = μ₋ₖₖ(k, x, Σ)
    
    if prod(size(Σ₋ₖₖ_)) == 1
        logcdf_part = logsubexp(
            logcdf(Normal(μ₋ₖₖ_[1],√(Σ₋ₖₖ_[1,1])), b[Not(k)][1]),
            logcdf(Normal(μ₋ₖₖ_[1],√(Σ₋ₖₖ_[1,1])), a[Not(k)][1]))
    elseif prod(size(Σ₋ₖₖ_)) == 0
        logcdf_part = 0.0
    else
        logcdf_part = mvnormcdf(μ₋ₖₖ_,Σ₋ₖₖ_,a[Not(k)],b[Not(k)])[1] |> log
    end
    logpdf_part = logpdf(Normal(zero(x), √(Σ[k,k])), x)
    (logpdf_part + logcdf_part) |> exp
end

function F(kq::Vector{Int}, x::Vector{T}, Σ::AbstractMatrix, a::AbstractVector{T}, b::AbstractVector{T}) where {T <: Real}
    Σ₋ₖₖ_ = Σ₋ₖₖ(kq, Σ)
    μ₋ₖₖ_ = μ₋ₖₖ(kq, x, Σ)
    
    if prod(size(Σ₋ₖₖ_)) == 1
        logcdf_part = logsubexp(
            logcdf(Normal(μ₋ₖₖ_[1],√(Σ₋ₖₖ_[1,1])), b[Not(kq)][1]),
            logcdf(Normal(μ₋ₖₖ_[1],√(Σ₋ₖₖ_[1,1])), a[Not(kq)][1]))
    elseif prod(size(Σ₋ₖₖ_)) == 0
        logcdf_part = 0.0
    else
        logcdf_part = mvnormcdf(μ₋ₖₖ_,Σ₋ₖₖ_,a[Not(kq)],b[Not(kq)])[1] |> log
    end
    logpdf_part = logpdf(MvNormal(zero(x), Σ[kq,kq]), x)
    (logpdf_part + logcdf_part) |> exp
end

F(k::Int, x::Real, d::TruncatedMvNormal) = F(k, x, d.normal.Σ, d.a, d.b)

const NORMALDIST = Normal(0,1)
normcdf(z) = cdf(NORMALDIST, z)

function α(Σ,a,b)
    if length(a) > 1
        return mvnormcdf(Σ, a ,b)[1]
    else
        return normcdf(b[1]/sqrt(Σ[1,1])) - normcdf(a[1]/sqrt(Σ[1,1]))
    end
end

EXᵢ(Σ,a,b) = (Σ * [F(k, a[k],Σ,a,b)-F(k,b[k],Σ,a,b) for k ∈ 1:length(a)]) ./ α(Σ,a,b)

function EXᵢ!(M1,throwaway, Σ,a,b)
    prob_mass = α(Σ,a,b)
    for k ∈ 1:length(throwaway)
        throwaway[k] = (F(k, a[k],Σ,a,b)-F(k,b[k],Σ,a,b))/prob_mass
    end
    mul!(M1, Σ ,throwaway)
end

EYᵢ(d::TruncatedMvNormal) = EXᵢ(d.normal.Σ,d.a .- d.normal.μ, d.b .- d.normal.μ) .+ d.normal.μ

function EXᵢXⱼ!(M2, Σ,a,b)
    #M2 = zero(Σ)
    α0 = α(Σ,a,b)
    d = size(Σ,1)
    for i ∈ 1:d, j ∈ 1:d
        M2[i,j] = α0*Σ[i,j]
        for k ∈ 1:d
            con = (Σ[i,k]*Σ[j,k]/Σ[k,k])
            con *= (a[k]*F(k,a[k], Σ, a, b) - b[k]*F(k,b[k], Σ, a, b))
            M2[i,j] += con
            insidek = zero(Σ[i,j])
            for q ∈ 1:d
                if q != k
                    con1 = (Σ[j,q] - (Σ[k,q]*Σ[j,k]/Σ[k,k]))
                    con2 = F([k,q],[a[k],a[q]],Σ,a,b)
                    con2 += F([k,q],[b[k],b[q]],Σ,a,b)
                    con2 -= F([k,q],[a[k],b[q]],Σ,a,b)
                    con2 -= F([k,q],[b[k],a[q]],Σ,a,b)
                    insidek += con1*con2
                end
            end
            M2[i,j] += Σ[i,k]*insidek
        end
    end
    M2 ./= α0
end

function EXᵢXⱼ(Σ,a,b)
    M2 = zero(Σ)
    EXᵢXⱼ!(M2, Σ,a,b)
    M2
end
    
function moments(TN::TruncatedMvNormal{FullNormal,T}) where {T <: Real}
    EX1 = EXᵢ(TN.normal.Σ,TN.a .- TN.normal.μ, TN.b .- TN.normal.μ)
    EX2 = EXᵢXⱼ(TN.normal.Σ, TN.a.-TN.normal.μ, TN.b.-TN.normal.μ)
    EY1 = EX1 .+ TN.normal.μ
    EY2 = EX2 .+ TN.normal.μ * EX1' .+ EX1 * TN.normal.μ' .+ TN.normal.μ * TN.normal.μ'
    #EY2 = EY1 * EY1' .+ EX2 .- EX1 * EX1'
    EY1, EY2
end


function moments(TN::TruncatedMvNormal{FullNormal,T}, block_structure::Vector{L}) where {T <: Real, L}
    EX1 = EXᵢ(TN.normal.Σ,TN.a .- TN.normal.μ, TN.b .- TN.normal.μ)
    EX2 = EXᵢXⱼ(TN.normal.Σ, TN.a.-TN.normal.μ, TN.b.-TN.normal.μ)
    EY1 = EX1 .+ TN.normal.μ
    EY2 = EX2 .+ TN.normal.μ * EX1' .+ EX1 * TN.normal.μ' .+ TN.normal.μ * TN.normal.μ'
    #EY2 = EY1 * EY1' .+ EX2 .- EX1 * EX1'
    EY1, EY2
end


function EXᵢXⱼ!(M1s, M2s, throwaway, Σ, a, b, bs::CovarianceBlockStructure)
    for (l,i_s) in enumerate(bs.block_indices)
        M1_i = @view(M1s[i_s])
        a_i = @view(a[i_s])
        b_i = @view(b[i_s])
        Σ_ii = @view(Σ[i_s, i_s])
        throwaway_i = @view(throwaway[i_s])
        EXᵢ!(M1_i,throwaway_i, Σ_ii,a_i,b_i)
    end
    
    for (l,i_s) in enumerate(bs.block_indices)
        M1_i = @view(M1s[i_s])
        μ_i = @view(a[i_s])
        a_i = @view(a[i_s])
        b_i = @view(b[i_s])
        for (k,j_s) in enumerate(bs.block_indices)
            M2_ij = @view(M2s[i_s, j_s])
            Σ_ij = @view(Σ[i_s, j_s])
            M1_j = @view(M1s[j_s])
            if l == k
                ### Within the block
                EXᵢXⱼ!(M2_ij, Σ_ij, a_i, b_i)
            else
                ### Off Block element
                for i in i_s
                    for j in j_s
                        M2s[i,j] = M1s[i]*M1s[j]
                    end
                end
            end
        end
    end
end

function EXᵢXⱼ(Σ, a, b, bs::CovarianceBlockStructure)
    M2s = zeros(eltype(Σ), size(Σ))
    M1s = zeros(eltype(a), size(a))
    throwaway = zeros(eltype(a), size(a))
    EXᵢXⱼ!(M1s, M2s, throwaway, Σ, a, b, bs)
    M2s
end

function moments!(M1s, M2s, throwaway, μ::AbstractVector, Σ::AbstractArray, a, b, bs::CovarianceBlockStructure)
    EXᵢXⱼ!(M1s, M2s, throwaway, Σ, a.-μ, b.-μ, bs)
    
    EY1 = M1s .+ μ
    EY2 = M2s .+ μ * M1s' .+ M1s * μ' .+ μ * μ'
    return EY1, EY2
end

function moments(μ::AbstractVector, Σ::AbstractArray, a::AbstractVector, b::AbstractVector, bs::CovarianceBlockStructure)
    M2s = zeros(size(Σ))
    M1s = zeros(size(a))
    throwaway = zeros(size(a))
    EXᵢXⱼ!(M1s, M2s, throwaway, Σ, a.-μ, b.-μ, bs)
    
    EY1 = M1s .+ μ
    EY2 = M2s .+ μ * M1s' .+ M1s * μ' .+ μ * μ'
    return EY1, EY2
end












