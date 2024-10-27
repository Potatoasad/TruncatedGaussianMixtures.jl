# Transformations in TGMM Fitting

This section describes how to use transformations to modify data in a way that aids in fitting Truncated Gaussian Mixture Models (TGMM) by changing the coordinate system.

## AbstractTransformation

`AbstractTransformation` is an abstract type that provides a blueprint for creating coordinate transformations on `DataFrame`s. Transformations can modify how data is projected, helping to improve TGMM fitting performance by enabling users to operate in different coordinate systems.


## Transformation Struct
Transformation is a concrete type implementing AbstractTransformation, allowing users to define custom transformations. Users can specify which columns represent the domain and image spaces and define functions for the forward and inverse transformations.

## Fields
`domain_columns`: Columns representing the domain of the transformation.
`forward`: The transformation function from domain to image.
`image_columns`: Columns in the transformed space.
`inverse`: The inverse transformation function.
`ignore_columns`: Columns that remain unchanged during transformations.

## Example Usage
```julia
using DataFrames

df_cartesian = DataFrame(:x => [1.0, 3.0], :y => [0.0, 1.0], :label => [:cat, :dog])

# Define a Cartesian-to-Polar transformation
CartesianToPolar = Transformation(
    [:x, :y],
    (x, y) -> (√(x^2 + y^2), atan2(y, x)),
    [:r, :θ],
    (r, θ) -> (r * cos(θ), r * sin(θ)),
    [:label]
)

# Apply forward transformation
df_polar = forward(CartesianToPolar, df_cartesian)

# Apply inverse transformation
df_cartesian_reconstructed = inverse(CartesianToPolar, df_polar)
```


## Misc Documentation

```@autodocs
Modules = [TruncatedGaussianMixtures]
Order = [:type, :function]
Private = false
```