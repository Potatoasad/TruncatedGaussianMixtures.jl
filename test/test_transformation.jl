using Revise, TruncatedGaussianMixtures, DataFrames

df = DataFrame(Dict(["x" => rand(1000), "y" => rand(1000)]));
T = Transformation(["x", "y"], (x,y)->(x^2, y), ["x2", "y"], (x2,y)->(x2^(0.5), y));

#@show T
T2 = add_quantile_transformation(T, df);
#@show T2

@show df
@show inverse(T2, forward(T2,df))