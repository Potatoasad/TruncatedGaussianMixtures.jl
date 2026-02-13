struct QuantileTransformer
    quantiles::Vector{Float64}
    references::Vector{Float64}
    output_distribution::Symbol
    a::Float64
    b::Float64
end

get_lower_bound(X::Vector{T}, a) where {T <: Number} = minimum(X)
get_lower_bound(X::Vector{T}, a::Number) where {T <: Number} = convert(T, a)

get_upper_bound(X::Vector{T}, b) where {T <: Number} = maximum(X)
get_upper_bound(X::Vector{T}, b::Number) where {T <: Number} = convert(T, b)

function fit_quantile_transformer(X::Vector; a=nothing, b=nothing, n_quantiles::Int=1000, output_distribution::Symbol=:uniform, ϵ=1e-10)
    # Clip number of quantiles to data size
    n_quantiles = min(n_quantiles, length(X))

    # Compute empirical quantiles
    quantiles = quantile(X, range(ϵ, 1-ϵ, length=n_quantiles))

    # Generate corresponding reference values
    if output_distribution == :uniform
        references = range(ϵ, 1-ϵ, length=n_quantiles)
    elseif output_distribution == :normal
        references = quantile(Normal(), range(ϵ, 1-ϵ, length=n_quantiles))
    else
        error("Invalid output distribution. Use :uniform or :normal.")
    end

    a,b = get_lower_bound(X, a), get_upper_bound(X, b)

    return QuantileTransformer(quantiles, references, output_distribution, a, b)
end

function transform(quantile_transformer::QuantileTransformer, X::Vector{Float64})
    # Interpolate values based on learned quantiles
    itp = linear_interpolation(quantile_transformer.quantiles, quantile_transformer.references, extrapolation_bc=Line())
    return itp.(X)
end

function transform(quantile_transformer::QuantileTransformer, X::Number)
    # Interpolate values based on learned quantiles
    itp = linear_interpolation(quantile_transformer.quantiles, quantile_transformer.references, extrapolation_bc=Line())
    return itp(X)
end

function inverse_transform(quantile_transformer::QuantileTransformer, X::Vector{Float64})
    # Reverse interpolation
    itp = linear_interpolation(quantile_transformer.references, quantile_transformer.quantiles, extrapolation_bc=Line())
    return clamp.(itp.(X), quantile_transformer.a, quantile_transformer.b)
end

function inverse_transform(quantile_transformer::QuantileTransformer, X::Number)
    # Reverse interpolation
    itp = linear_interpolation(quantile_transformer.references, quantile_transformer.quantiles, extrapolation_bc=Line())
    return clamp(itp(X), quantile_transformer.a, quantile_transformer.b)
end

add_quantile_suffix(x::String) = "$(x)_quantile"
add_quantile_suffix(x::Symbol) = Symbol("$(string(x))_quantile")

function add_quantile_transformation(T::AbstractTransformation, df::DataFrame; ignore_quantile_columns=[], output_distribution=:normal, n_quantiles=100)
    columns = image_columns(T)
    df_forward = forward(T, df)
    qts = [fit_quantile_transformer(df_forward[!, col], n_quantiles=n_quantiles, output_distribution=output_distribution) for col in columns];
    
    function forward_transform_quantile_only(θ...)
        θ_transformed = zeros(length(θ))
        for (i,x) in enumerate(θ)
            if i ∉ ignore_quantile_columns#[3,4,5,6]
                θ_transformed[i] = transform(qts[i], x)
            else
                θ_transformed[i] = x
            end
        end
        tuple(θ_transformed...)
    end

    function inverse_transform_quantile_only(θ_transformed...)
        θ = zeros(length(θ_transformed))
        for (i,x) in enumerate(θ_transformed)
            if i ∉ ignore_quantile_columns#[3,4,5,6]
                θ[i] = inverse_transform(qts[i], x)
            else
                θ[i] = x
            end
        end
        tuple(θ...)
    end

    internal_quantile_transformation = Transformation(
        T.image_columns,
        forward_transform_quantile_only,
        [i ∉ ignore_quantile_columns ? add_quantile_suffix(col) : col for (i,col) in enumerate(T.image_columns)],
        inverse_transform_quantile_only,
        T.ignore_columns
    )

    compose(T, internal_quantile_transformation)
end




