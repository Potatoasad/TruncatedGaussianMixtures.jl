using Test
using SHA
using DataFrames
using Distributions
using LinearAlgebra
using TruncatedGaussianMixtures

function file_sha256(path::AbstractString)
    bytes2hex(sha256(read(path)))
end

function fixture_path()
    candidates = [
        joinpath(@__DIR__, "fixtures", "fit_sample.hdf5"),
        joinpath(@__DIR__, "fixtures", "fit.hdf5"),
        joinpath(@__DIR__, "fit.hdf5"),
    ]
    for p in candidates
        if isfile(p)
            return p
        end
    end
    return nothing
end

function build_simple_gmm()
    a = [0.0, 0.0]
    b = [1.0, 1.0]
    μ1 = [0.2, 0.3]
    μ2 = [0.7, 0.8]
    Σ1 = [0.03 0.01; 0.01 0.04]
    Σ2 = [0.05 0.00; 0.00 0.02]
    mix = MixtureModel(
        [TruncatedMvNormal(MvNormal(μ1, Σ1), a, b), TruncatedMvNormal(MvNormal(μ2, Σ2), a, b)],
        [0.4, 0.6],
    )
    return mix
end

@testset "TGMM HDF5 Interop" begin
    fixture = fixture_path()
    @test !isnothing(fixture)

    if isnothing(fixture)
        return
    end

    @testset "Load fixture" begin
        fit = load_tgmm_h5(fixture)
        @test fit isa TGMMFit
        @test length(fit.gmm.components) > 0
        @test length(fit.cols) == length(fit.gmm.components[1].normal.μ)
    end

    @testset "No-op save keeps bytes identical" begin
        sha_before = file_sha256(fixture)
        size_before = filesize(fixture)

        fit = load_tgmm_h5(fixture)
        save_tgmm_h5(fixture, fit)

        sha_after = file_sha256(fixture)
        size_after = filesize(fixture)
        @test sha_after == sha_before
        @test size_after == size_before
    end

    @testset "Mutation rewrites file" begin
        tmpdir = mktempdir()
        path = joinpath(tmpdir, "fit.hdf5")
        cp(fixture, path; force=true)

        sha_before = file_sha256(path)
        fit = load_tgmm_h5(path)
        if isnothing(fit.cov)
            fit.cov = "full"
        else
            fit.cov = fit.cov == "full" ? "diag" : "full"
        end
        save_tgmm_h5(path, fit)

        sha_after = file_sha256(path)
        @test sha_after != sha_before

        fit2 = load_tgmm_h5(path)
        @test fit2.cov == fit.cov
    end

    @testset "Unknown extras round-trip" begin
        tmpdir = mktempdir()
        path = joinpath(tmpdir, "fit.hdf5")
        cp(fixture, path; force=true)

        fit = load_tgmm_h5(path)
        fit.extras["julia_extra"] = Dict("alpha" => [1, 2, 3], "beta" => "ok")
        save_tgmm_h5(path, fit)

        fit2 = load_tgmm_h5(path)
        @test haskey(fit2.extras, "julia_extra")
        @test fit2.extras["julia_extra"]["beta"] == "ok"
    end

    @testset "Julia-origin fit and null transformation" begin
        gmm = build_simple_gmm()
        df = DataFrame(x=[0.1, 0.2, 0.9], y=[0.2, 0.6, 0.8])
        responsibilities = [0.9 0.1; 0.7 0.3; 0.2 0.8]
        fit = TGMMFit(
            gmm;
            cols=["x", "y"],
            domain_cols=["x", "y"],
            image_cols=["x", "y"],
            responsibilities=responsibilities,
            block_structure=[0, 0],
            cov="full",
            data=df,
        )

        tmpdir = mktempdir()
        path = joinpath(tmpdir, "julia_fit.hdf5")
        save_tgmm_h5(path, fit)

        fit2 = load_tgmm_h5(path)
        @test isnothing(fit2.transformation)
        @test fit2.cols == ["x", "y"]
        @test size(fit2.responsibilities) == (3, 2)
    end

    @testset "Transformation override" begin
        gmm = build_simple_gmm()
        fit = TGMMFit(gmm; cols=["x", "y"], domain_cols=["x", "y"], image_cols=["x", "y"])
        override = Dict(
            "input_columns" => ["x", "y"],
            "forward_transformation" => "(x,y)->(x,y)",
            "transformed_columns" => ["x", "y"],
            "inverse_transformation" => "(x,y)->(x,y)",
            "ignore_columns" => String[],
            "extra_funcs" => nothing,
            "quantile_transformation" => nothing,
        )

        tmpdir = mktempdir()
        path = joinpath(tmpdir, "override_fit.hdf5")
        save_tgmm_h5(path, fit; transformation_override=override)

        fit2 = load_tgmm_h5(path)
        @test !isnothing(fit2.transformation)
        @test fit2.transformation["forward_transformation"] == "(x,y)->(x,y)"
    end
end
