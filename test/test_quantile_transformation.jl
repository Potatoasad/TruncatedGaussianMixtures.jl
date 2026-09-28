# The quantile transformer builds its interpolants once; results must equal a
# freshly built interpolation (the previous per-value implementation).
using TruncatedGaussianMixtures, Random, Distributions, Test
using TruncatedGaussianMixtures.Interpolations: linear_interpolation, Line
const TGMMq = TruncatedGaussianMixtures

@testset "quantile transformer" begin
    Random.seed!(2)
    for od in (:uniform, :normal)
        X = vcat(rand(Gamma(2, 3), 5000), fill(4.0, 20))          # repeated knots
        qt = TGMMq.fit_quantile_transformer(X; n_quantiles=length(X), output_distribution=od)
        f = linear_interpolation(copy(qt.quantiles), copy(qt.references), extrapolation_bc=Line())
        g = linear_interpolation(copy(qt.references), copy(qt.quantiles), extrapolation_bc=Line())
        Y = vcat(rand(Gamma(2, 3), 500), [-1.0, 100.0])            # extrapolation
        @test [TGMMq.transform(qt, y) for y in Y] == f.(Y)
        @test TGMMq.transform(qt, Y) == f.(Y)
        Z = TGMMq.transform(qt, Y)
        @test [TGMMq.inverse_transform(qt, z) for z in Z] == clamp.(g.(Z), qt.a, qt.b)
        # per-value calls are O(1) in the number of quantiles now
        @test (@elapsed [TGMMq.transform(qt, x) for x in X]) < 0.5
    end
end
