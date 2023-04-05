
using Statistics: mean, cov

function μ₋ₖₖ(k::Int, x::Real, Σ::AbstractMatrix)
    Σ[Not(k),k] * inv(Σ[k,k]) * x
end

function μ₋ₖₖ(kq::Vector{Int}, x::Vector{T}, Σ::AbstractMatrix) where {T <: Real}
    Σ[Not(kq),kq] * inv(Σ[kq,kq]) * x
end
    
function Σ₋ₖₖ(k::Int, Σ::AbstractMatrix)
    Σ[Not(k),Not(k)] .- (Σ[Not(k),k] * inv(Σ[k,k]) * Σ[k, Not(k)]')
end

function Σ₋ₖₖ(kq::Vector{Int}, Σ::AbstractMatrix)
    Σ[Not(kq),Not(kq)] .- (Σ[Not(kq),kq] * inv(Σ[kq,kq]) * Σ[kq, Not(kq)])
end

#√ᵪ(x) = exp(logsumexp([log(x),log(10*eps(x))])*(1/2));
√ᵪ(x) = √(x);

function F(k::Int, x::Real, Σ::AbstractMatrix, a::Vector{T}, b::Vector{T}) where {T <: Real}
    Σ₋ₖₖ_ = Σ₋ₖₖ(k, Σ)
    μ₋ₖₖ_ = μ₋ₖₖ(k, x, Σ)
    
    if prod(size(Σ₋ₖₖ_)) == 1
        logcdf_part = logsubexp(
            logcdf(Normal(μ₋ₖₖ_[1],√ᵪ(Σ₋ₖₖ_[1,1])), b[Not(k)][1]),
            logcdf(Normal(μ₋ₖₖ_[1],√ᵪ(Σ₋ₖₖ_[1,1])), a[Not(k)][1]))
    elseif prod(size(Σ₋ₖₖ_)) == 0
        logcdf_part = 0.0
    else
        logcdf_part = mvnormcdf(μ₋ₖₖ_,Σ₋ₖₖ_,a[Not(k)],b[Not(k)])[1] |> log
    end
    logpdf_part = logpdf(Normal(zero(x), √ᵪ(Σ[k,k])), x)
    (logpdf_part + logcdf_part) |> exp
end

function F(kq::Vector{Int}, x::Vector{T}, Σ::AbstractMatrix, a::Vector{T}, b::Vector{T}) where {T <: Real}
    Σ₋ₖₖ_ = Σ₋ₖₖ(kq, Σ)
    μ₋ₖₖ_ = μ₋ₖₖ(kq, x, Σ)
    
    if prod(size(Σ₋ₖₖ_)) == 1
        logcdf_part = logsubexp(
            logcdf(Normal(μ₋ₖₖ_[1],√ᵪ(Σ₋ₖₖ_[1,1])), b[Not(kq)][1]),
            logcdf(Normal(μ₋ₖₖ_[1],√ᵪ(Σ₋ₖₖ_[1,1])), a[Not(kq)][1]))
    elseif prod(size(Σ₋ₖₖ_)) == 0
        logcdf_part = 0.0
    else
        logcdf_part = mvnormcdf(μ₋ₖₖ_,Σ₋ₖₖ_,a[Not(kq)],b[Not(kq)])[1] |> log
    end
    logpdf_part = logpdf(MvNormal(zero(x), Σ[kq,kq]), x)
    (logpdf_part + logcdf_part) |> exp
end

F(k::Int, x::Real, d::TruncatedMvNormal) = F(k, x, d.normal.Σ, d.a, d.b)

α(Σ,a,b) = mvnormcdf(zero(size(Σ,2)), Σ, a ,b)[1]

EXᵢ(Σ,a,b) = (Σ * [F(k, a[k],Σ,a,b)-F(k,b[k],Σ,a,b) for k ∈ 1:length(a)]) ./ α(Σ,a,b)

EYᵢ(d::TruncatedMvNormal) = EXᵢ(d.normal.Σ,d.a .- d.normal.μ, d.b .- d.normal.μ) .+ d.normal.μ

function EXᵢXⱼ(Σ,a,b)
    M2 = zero(Σ)
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
    (M2 ./ α0)
end
    
function moments(TN::TruncatedMvNormal{FullNormal,T}) where {T <: Real}
    EX1 = EXᵢ(TN.normal.Σ,TN.a .- TN.normal.μ, TN.b .- TN.normal.μ)
    EX2 = EXᵢXⱼ(TN.normal.Σ, TN.a.-TN.normal.μ, TN.b.-TN.normal.μ)
    EY1 = EX1 .+ TN.normal.μ
    EY2 = EX2 .+ TN.normal.μ * EX1' .+ EX1 * TN.normal.μ' .+ TN.normal.μ * TN.normal.μ'
    #EY2 = EY1 * EY1' .+ EX2 .- EX1 * EX1'
    EY1, EY2
end
