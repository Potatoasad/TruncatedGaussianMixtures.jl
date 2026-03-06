using Random
using LinearAlgebra
using Distributions
using Plots

# If these are from your package, keep the import; otherwise remove/change it.
using TruncatedGaussianMixtures

# -----------------------------
# Build the 2D (indices 3,4) sector
# -----------------------------
μ4 = [0.4231732131275458, -0.5044912531304235, -0.9647385179161311, 6.468569875628318]

Σ4 = [
     0.05481154092002593  -0.3323922791905646   0.0                 0.0;
    -0.3323922791905646    2.3501858547019836   0.0                 0.0;
     0.0                   0.0                  1.325240390514753  -4.318957814403392;
     0.0                   0.0                 -4.318957814403392  14.823432555725496
]

a4 = [0.0, 0.0, -1.0, -1.0]
b4 = [1.0, 1.0,  1.0,  1.0]

inds = 3:4
μ = μ4[inds]
Σ = [1.325240390514753  -4.318957814403392; -4.318957814403392  14.823432555725496]; #Σ4[inds, inds]
a = a4[inds]
b = b4[inds]

normal2 = MvNormal(μ, Σ)
d2 = TruncatedMvNormal(normal2, a, b)




rand(normal2, 10_000_000)


# -----------------------------
# Diagonal proposal on the same box
# -----------------------------
function diagnormal_dists(d)
    [truncated(Normal(μ, sqrt(Σii)), ak, bk)
     for (μ, Σii, ak, bk) in zip(d.normal.μ, diag(d.normal.Σ), d.a, d.b)]
end

function sample_diag_trunc(rng, d, N)
    dists = diagnormal_dists(d)
    D = length(d)
    X = Matrix{Float64}(undef, D, N)
    for i in 1:N
        for k in 1:D
            X[k, i] = rand(rng, dists[k])
        end
    end
    return X
end

function logpdf_diag_proposal(d, x::AbstractVector)
    s = 0.0
    for k in 1:length(d)
        s += logpdf(Normal(d.normal.μ[k], sqrt(d.normal.Σ[k,k])), x[k])
    end
    return s
end

# -----------------------------
# Exact rejection sampler from the truncated target
# -----------------------------
function rejection_sample(rng, d, N)
    D = length(d)
    X = Matrix{Float64}(undef, D, N)
    x = Vector{Float64}(undef, D)
    i = 1
    while i <= N
        rand!(rng, d.normal, x)
        if insupport(d, x)
            X[:, i] .= x
            i += 1
        end
    end
    return X
end

# -----------------------------
# Main test
# -----------------------------
rng = MersenneTwister(1234)
N = 100_000

# "standard" = whatever rand(d, N) currently uses in your implementation
X_std = rand(d2, N)

# exact baseline
X_rej = rejection_sample(rng, d2, N)

# optional: diagonal proposal draws, if you also want to inspect proposal vs target pdf
X_prop = sample_diag_trunc(rng, d2, N)

# -----------------------------
# 1) Sample scatter comparison
# -----------------------------
p1 = scatter(
    X_std[1, :], X_std[2, :];
    ms=1.5, alpha=0.04, label="standard rand(d, N)",
    xlabel="x₃", ylabel="x₄", title="2D sector (indices 3,4): sample comparison"
)

scatter!(
    p1,
    X_rej[1, :], X_rej[2, :];
    ms=1.5, alpha=0.04, label="exact rejection"
)

# -----------------------------
# 2) PDF comparison scatter
#    Compare target truncated logpdf vs diagonal-proposal-based logpdf
#    on the same proposal sample cloud
# -----------------------------
logp_target = [logpdf(d2, view(X_prop, :, i)) for i in 1:N]
logp_diag   = [logpdf_diag_proposal(d2, view(X_prop, :, i)) for i in 1:N]

p2 = scatter(
    logp_diag, logp_target;
    ms=2, alpha=0.04, label="samples from diagonal proposal",
    xlabel="log q_diag(x)", ylabel="log p_trunc(x)",
    title="PDF comparison on common sample points"
)

plot(p1, p2, layout=(1,2), size=(1200, 500))