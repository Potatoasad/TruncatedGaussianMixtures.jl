# Abstract type for transformation operations on DataFrames
"""
An abstract type representing a coordinate transformation.

`AbstractTransformation` is a supertype for transformations that can be applied to a DataFrame.
It requires that any subtype define `forward`, `inverse`, `domain_columns`, and `image_columns` methods.

# Required methods for subtypes
- `forward`: Defines the forward transformation.
- `inverse`: Defines the inverse transformation.
- `domain_columns`: Lists columns in the DataFrame that belong to the transformation's domain.
- `image_columns`: Lists columns representing the transformed space.
"""
abstract type AbstractTransformation end

"""
Return the forward transformation function of the specified transformation.

# Arguments
- `Tr::AbstractTransformation`: A transformation object.

# Returns
- The forward transformation function associated with `Tr`.
"""
forward(Tr::AbstractTransformation) = Tr.forward

"""
Return the domain columns of the specified transformation.

# Arguments
- `Tr::AbstractTransformation`: A transformation object.

# Returns
- A list of symbols corresponding to the domain columns.
"""
domain_columns(Tr::AbstractTransformation) = Tr.domain_columns

"""
Return the inverse transformation function of the specified transformation.

# Arguments
- `Tr::AbstractTransformation`: A transformation object.

# Returns
- The inverse transformation function associated with `Tr`.
"""
inverse(Tr::AbstractTransformation) = Tr.inverse

"""
Return the image columns of the specified transformation.

# Arguments
- `Tr::AbstractTransformation`: A transformation object.

# Returns
- A list of symbols corresponding to the columns in the transformed space.
"""
image_columns(Tr::AbstractTransformation) = Tr.image_columns

"""
Apply the forward transformation to a DataFrame.

Transforms columns in the domain space to the image space, keeping other columns unchanged.

# Arguments
- `Tr::AbstractTransformation`: A transformation object.
- `df::DataFrame`: A DataFrame containing columns matching the domain.

# Returns
- A DataFrame with transformed columns in the image space.
"""
function forward(Tr::AbstractTransformation, df::DataFrame)
    F = forward(Tr)
    X = F.(Tuple(df[!, col] for col ∈ domain_columns(Tr))...)
    ys = image_columns(Tr)
    df_out = DataFrame(X, ys)
    for col ∈ Tr.ignore_columns
        df_out[!, col] = df[!, col]
    end
    df_out
end

"""
Apply the inverse transformation to a DataFrame.

Transforms columns in the image space back to the domain space, keeping other columns unchanged.

# Arguments
- `Tr::AbstractTransformation`: A transformation object.
- `df::DataFrame`: A DataFrame containing columns in the image space.

# Returns
- A DataFrame with transformed columns in the domain space.
"""
function inverse(Tr::AbstractTransformation, df::DataFrame)
    F = inverse(Tr)
    X = F.(Tuple(df[!, col] for col ∈ image_columns(Tr))...)
    ys = domain_columns(Tr)
    df_out = DataFrame(X, ys)
    for col ∈ Tr.ignore_columns
        df_out[!, col] = df[!, col]
    end
    df_out
end

"""
A concrete implementation of `AbstractTransformation` for performing custom transformations on DataFrames.

# Fields
- `domain_columns`: A vector of symbols representing the columns in the domain space.
- `forward`: A function that maps domain columns to image columns.
- `image_columns`: A vector of symbols representing the columns in the transformed space.
- `inverse`: A function that maps image columns back to domain columns.
- `ignore_columns`: A vector of columns to leave unchanged during transformation.

# Constructor
`Transformation(domain_columns, forward, image_columns, inverse; ignore_columns=[])`
"""
struct Transformation{A,B,C,D,L} <: AbstractTransformation
    domain_columns::A
    forward::B
    image_columns::C
    inverse::D
    ignore_columns::L
end

Transformation(domain_columns, forward, image_columns, inverse; ignore_columns=[]) = 
    Transformation(domain_columns, forward, image_columns, inverse, ignore_columns)



"""
Usage:

julia> df_cartesian = DataFrame(:x => [1.0, 3.0], :y => [0.0, 1.0], :label => [:cat, :dog]) # Make a dataframe
2×3 DataFrame
 Row │ x        y        label
     │ Float64  Float64  Symbol
─────┼──────────────────────────
   1 │     1.0      0.0  cat
   2 │     3.0      1.0  dog

julia> CartesianToPolar = Transformation(
	[:x, :y],
	(x,y) -> (√(x^2 + y^2), atan2(y,x)),
	[:r, :θ],
	(r,θ) -> (r*cos(θ), r*sin(θ)),
	[:label]
)
Transformation{Vector{Symbol}, var"#5#7", Vector{Symbol}, var"#6#8", Vector{Symbol}}([:x, :y], var"#5#7"(), [:r, :θ], var"#6#8"(), [:label])

julia> df_polar = forward(CartesianToPolar, df) # returns 
2×3 DataFrame
 Row │ r        θ         label
     │ Float64  Float64   Symbol
─────┼───────────────────────────
   1 │ 1.0      0.0       cat
   2 │ 3.16228  0.321751  dog

julia> inverse(CartesianToPolar, df_polar)
2×3 DataFrame
 Row │ x        y        label
     │ Float64  Float64  Symbol
─────┼──────────────────────────
   1 │     1.0      0.0  cat
   2 │     3.0      1.0  dog

 julia> inverse(CartesianToPolar, df_polar) .== df
2×3 DataFrame
 Row │ x     y     label
     │ Bool  Bool  Bool
─────┼───────────────────
   1 │ true  true   true
   2 │ true  true   true
"""

