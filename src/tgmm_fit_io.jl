const _NONE_ATTR = "__is_none__"

using HDF5
using SHA

mutable struct TGMMFit
    gmm::MixtureModel
    cols::Vector{String}
    domain_cols::Vector{String}
    image_cols::Vector{String}
    responsibilities::Union{Nothing,Matrix{Float64}}
    block_structure::Union{Nothing,Vector{Int}}
    cov::Union{Nothing,String}
    data::Union{Nothing,DataFrame}
    components::Union{Nothing,Vector{Int}}
    sample_weights::Any
    transformation::Union{Nothing,Dict{String,Any}}
    extras::Dict{String,Any}
    _source_path::Union{Nothing,String}
    _source_file_sha256::Union{Nothing,String}
    _loaded_dict_snapshot::Union{Nothing,Dict{String,Any}}
end

function TGMMFit(
    gmm::MixtureModel;
    cols=String[],
    domain_cols=String[],
    image_cols=String[],
    responsibilities=nothing,
    block_structure=nothing,
    cov=nothing,
    data=nothing,
    components=nothing,
    sample_weights=nothing,
    transformation=nothing,
    extras=Dict{String,Any}(),
    source_path=nothing,
    source_file_sha256=nothing,
    loaded_dict_snapshot=nothing,
)
    responsibilities_matrix = _normalize_responsibilities(responsibilities)
    block_structure_vec = _normalize_int_vector(block_structure)
    components_vec = _normalize_int_vector(components)
    transformation_dict = _normalize_string_any_dict(transformation)

    TGMMFit(
        gmm,
        _string_vector(cols),
        _string_vector(domain_cols),
        _string_vector(image_cols),
        responsibilities_matrix,
        block_structure_vec,
        _maybe_string(cov),
        _normalize_dataframe(data),
        components_vec,
        sample_weights,
        transformation_dict,
        _normalize_string_any_dict(extras, default=Dict{String,Any}()),
        isnothing(source_path) ? nothing : abspath(String(source_path)),
        isnothing(source_file_sha256) ? nothing : String(source_file_sha256),
        isnothing(loaded_dict_snapshot) ? nothing : deepcopy(_normalize_string_any_dict(loaded_dict_snapshot)),
    )
end

function _normalize_string_any_dict(x; default=nothing)
    if isnothing(x)
        return default
    end
    x isa Dict || throw(ArgumentError("Expected Dict-like object, got $(typeof(x))"))
    out = Dict{String,Any}()
    for (k, v) in x
        out[String(k)] = v
    end
    out
end

_string_vector(x::AbstractVector) = String[String(v) for v in x]
_string_vector(::Nothing) = String[]

_maybe_string(::Nothing) = nothing
_maybe_string(x) = String(x)

function _normalize_int_vector(::Nothing)
    return nothing
end

function _normalize_int_vector(v)
    [Int(x) for x in v]
end

function _normalize_responsibilities(::Nothing)
    return nothing
end

function _normalize_responsibilities(x)
    arr = Float64.(Array(x))
    if ndims(arr) == 1
        return reshape(arr, :, 1)
    elseif ndims(arr) == 2
        return arr
    end
    throw(ArgumentError("responsibilities must be 1D or 2D"))
end

_normalize_dataframe(::Nothing) = nothing

function _normalize_dataframe(df)
    df isa DataFrame && return copy(df)
    return DataFrame(df)
end

function _compute_sha256(path::AbstractString)
    bytes = read(path)
    bytes2hex(sha256(bytes))
end

function _to_data_matrix(df::DataFrame)
    Matrix(df)
end

function _scalar_or_array_to_julia(x)
    if x isa AbstractArray
        return x
    end
    return x
end

function _dataset_to_value(ds)
    x = read(ds)
    if x isa AbstractArray && ndims(x) == 0
        return x[]
    end
    if x isa AbstractString
        return String(x)
    elseif x isa AbstractVector{UInt8}
        return String(x)
    elseif x isa AbstractArray{<:AbstractString}
        return String.(x)
    elseif x isa AbstractArray
        return _scalar_or_array_to_julia(x)
    end
    return x
end

function _load_group(group)::Dict{String,Any}
    out = Dict{String,Any}()
    for name in keys(group)
        obj = group[name]
        key = String(name)
        if obj isa HDF5.Group
            attrs_obj = attrs(obj)
            if haskey(attrs_obj, _NONE_ATTR) && Bool(attrs_obj[_NONE_ATTR])
                out[key] = nothing
            elseif haskey(obj, "columns") && haskey(obj, "values")
                raw_cols = _dataset_to_value(obj["columns"])
                cols = raw_cols isa AbstractVector ? String[String(c) for c in raw_cols] : String[String(raw_cols)]
                values = read(obj["values"])
                if ndims(values) != 2
                    throw(ArgumentError("Expected DataFrame values dataset to be 2D, got ndims=$(ndims(values))"))
                end
                if size(values, 2) == length(cols)
                    out[key] = DataFrame(values, Symbol.(cols))
                elseif size(values, 1) == length(cols)
                    out[key] = DataFrame(permutedims(values), Symbol.(cols))
                else
                    throw(DimensionMismatch("DataFrame values shape $(size(values)) incompatible with $(length(cols)) columns"))
                end
            else
                out[key] = _load_group(obj)
            end
        else
            out[key] = _dataset_to_value(obj)
        end
    end
    out
end

function _save_dataset!(group, key::AbstractString, value)
    group[String(key)] = value
end

function _is_string_collection(v)
    v isa AbstractVector || return false
    all(x -> x isa AbstractString, v)
end

function _save_group!(group, d::Dict{String,Any})
    for (k, v) in d
        key = String(k)
        if isnothing(v)
            g = create_group(group, key)
            attrs(g)[_NONE_ATTR] = true
        elseif v isa Dict
            _save_group!(create_group(group, key), Dict{String,Any}(String(kk) => vv for (kk, vv) in v))
        elseif v isa DataFrame
            g = create_group(group, key)
            _save_dataset!(g, "columns", String.(names(v)))
            _save_dataset!(g, "values", _to_data_matrix(v))
        elseif v isa Tuple
            _save_group!(group, Dict(key => collect(v)))
        elseif v isa AbstractString
            _save_dataset!(group, key, String(v))
        elseif _is_string_collection(v)
            _save_dataset!(group, key, String[String(x) for x in v])
        elseif v isa AbstractArray
            _save_dataset!(group, key, v)
        elseif v isa Number || v isa Bool
            _save_dataset!(group, key, v)
        else
            _save_dataset!(group, key, v)
        end
    end
end

function save_dict_h5(path::AbstractString, d::Dict{String,Any})
    h5open(path, "w") do f
        _save_group!(f, d)
    end
    nothing
end

function load_dict_h5(path::AbstractString)::Dict{String,Any}
    h5open(path, "r") do f
        return _load_group(f)
    end
end

function _get_required(d::Dict{String,Any}, k::String)
    haskey(d, k) || throw(ArgumentError("Missing required key '$k'"))
    d[k]
end

function _gmm_to_dict(gmm::MixtureModel)
    K = length(gmm.components)
    K > 0 || throw(ArgumentError("Mixture must have at least one component"))
    d = length(gmm.components[1].normal.μ)

    means = zeros(Float64, K, d)
    covariances = zeros(Float64, K, d, d)
    for k in 1:K
        means[k, :] .= Float64.(gmm.components[k].normal.μ)
        covariances[k, :, :] .= Float64.(Matrix(gmm.components[k].normal.Σ))
    end

    Dict{String,Any}(
        "means" => means,
        "covariances" => covariances,
        "weights" => Float64.(collect(gmm.prior.p)),
        "a" => Float64.(collect(gmm.components[1].a)),
        "b" => Float64.(collect(gmm.components[1].b)),
    )
end

function _gmm_from_dict(d::Dict{String,Any})
    means = Float64.(Array(_get_required(d, "means")))
    covariances = Float64.(Array(_get_required(d, "covariances")))
    weights = Float64.(vec(Array(_get_required(d, "weights"))))
    a = Float64.(vec(Array(_get_required(d, "a"))))
    b = Float64.(vec(Array(_get_required(d, "b"))))

    ndims(means) == 2 || throw(ArgumentError("gmm.means must be 2D (K x d)"))
    ndims(covariances) == 3 || throw(ArgumentError("gmm.covariances must be 3D (K x d x d)"))

    K = length(weights)
    means_kd = nothing
    cov_kdd = nothing

    if size(means, 1) == K && size(covariances, 1) == K
        # Python wrapper convention: means[K, d], covariances[K, d, d]
        d_means = size(means, 2)
        if size(covariances, 2) != d_means || size(covariances, 3) != d_means
            throw(ArgumentError("gmm.covariances dim mismatch"))
        end
        means_kd = means
        cov_kdd = covariances
    elseif size(means, 2) == K && size(covariances, 3) == K
        # Legacy convention: means[d, K], covariances[d, d, K]
        d_means = size(means, 1)
        if size(covariances, 1) != d_means || size(covariances, 2) != d_means
            throw(ArgumentError("gmm.covariances dim mismatch"))
        end
        means_kd = permutedims(means, (2, 1))
        cov_kdd = permutedims(covariances, (3, 1, 2))
    else
        throw(ArgumentError("Unsupported gmm means/covariances shape combination"))
    end

    comps = Vector{TruncatedMvNormal}(undef, K)
    for k in 1:K
        Σ = Matrix(cov_kdd[k, :, :])
        Σ = (Σ + Σ') ./ 2
        comps[k] = TruncatedMvNormal(MvNormal(vec(means_kd[k, :]), Σ), a, b)
    end
    MixtureModel(comps, weights)
end

function tgmmfit_from_dict(d::Dict{String,Any}; source_path=nothing, source_file_sha256=nothing)
    known = Set([
        "cols", "domain_cols", "image_cols", "responsibilities", "block_structure", "cov", "data",
        "gmm", "transformation", "components", "sample_weights",
    ])

    gmm_dict = _normalize_string_any_dict(_get_required(d, "gmm"))
    gmm = _gmm_from_dict(gmm_dict)

    cols = haskey(d, "cols") ? _string_vector(d["cols"]) : String[]
    domain_cols = haskey(d, "domain_cols") ? _string_vector(d["domain_cols"]) : copy(cols)
    image_cols = haskey(d, "image_cols") ? _string_vector(d["image_cols"]) : copy(cols)

    extras = Dict{String,Any}()
    for (k, v) in d
        if !(k in known)
            extras[k] = v
        end
    end

    fit = TGMMFit(
        gmm;
        cols=cols,
        domain_cols=domain_cols,
        image_cols=image_cols,
        responsibilities=get(d, "responsibilities", nothing),
        block_structure=get(d, "block_structure", nothing),
        cov=get(d, "cov", nothing),
        data=get(d, "data", nothing),
        components=get(d, "components", nothing),
        sample_weights=get(d, "sample_weights", nothing),
        transformation=get(d, "transformation", nothing),
        extras=extras,
        source_path=source_path,
        source_file_sha256=source_file_sha256,
        loaded_dict_snapshot=deepcopy(d),
    )

    return fit
end

function tgmmfit_to_dict(fit::TGMMFit; transformation_override=nothing, include_unknown=true)
    d = Dict{String,Any}(
        "cols" => copy(fit.cols),
        "domain_cols" => copy(fit.domain_cols),
        "image_cols" => copy(fit.image_cols),
        "responsibilities" => isnothing(fit.responsibilities) ? nothing : copy(fit.responsibilities),
        "block_structure" => isnothing(fit.block_structure) ? nothing : copy(fit.block_structure),
        "cov" => fit.cov,
        "data" => isnothing(fit.data) ? nothing : copy(fit.data),
        "gmm" => _gmm_to_dict(fit.gmm),
        "transformation" => isnothing(transformation_override) ? fit.transformation : _normalize_string_any_dict(transformation_override),
    )

    if !isnothing(fit.components)
        d["components"] = copy(fit.components)
    end
    if !isnothing(fit.sample_weights)
        d["sample_weights"] = fit.sample_weights
    end

    if include_unknown
        for (k, v) in fit.extras
            if !haskey(d, k)
                d[k] = deepcopy(v)
            end
        end
    end

    d
end

function _deep_equal(a, b)
    if a isa Dict && b isa Dict
        Set(keys(a)) == Set(keys(b)) || return false
        for k in keys(a)
            _deep_equal(a[k], b[k]) || return false
        end
        return true
    elseif a isa DataFrame && b isa DataFrame
        names(a) == names(b) || return false
        size(a) == size(b) || return false
        return all(isequal.(Matrix(a), Matrix(b)))
    elseif a isa AbstractArray && b isa AbstractArray
        size(a) == size(b) || return false
        return all(_deep_equal.(a, b))
    elseif a isa AbstractString && b isa AbstractString
        return String(a) == String(b)
    elseif a isa AbstractVector{UInt8} && b isa AbstractString
        return String(a) == String(b)
    elseif a isa AbstractString && b isa AbstractVector{UInt8}
        return String(a) == String(b)
    else
        return isequal(a, b)
    end
end

function _canonicalize_gmm_dict(gmm::Dict{String,Any})
    out = deepcopy(gmm)
    if haskey(out, "means") && haskey(out, "covariances") && haskey(out, "weights")
        means = Float64.(Array(out["means"]))
        covs = Float64.(Array(out["covariances"]))
        weights = Float64.(vec(Array(out["weights"])))
        K = length(weights)

        if ndims(means) == 2 && ndims(covs) == 3
            if size(means, 1) == K && size(covs, 1) == K
                out["means"] = means
                out["covariances"] = covs
            elseif size(means, 2) == K && size(covs, 3) == K
                out["means"] = permutedims(means, (2, 1))
                out["covariances"] = permutedims(covs, (3, 1, 2))
            end
        end
    end
    out
end

function _canonicalize_for_compare(x)
    if x isa Dict
        d = Dict{String,Any}()
        for (k, v) in x
            d[String(k)] = _canonicalize_for_compare(v)
        end
        if haskey(d, "gmm") && d["gmm"] isa Dict
            d["gmm"] = _canonicalize_gmm_dict(d["gmm"])
        end
        return d
    elseif x isa DataFrame
        return DataFrame(copy(x))
    elseif x isa AbstractArray
        return copy(x)
    end
    return x
end

function load_tgmm_h5(path::AbstractString)
    path_abs = abspath(path)
    d = load_dict_h5(path_abs)
    sha = _compute_sha256(path_abs)
    tgmmfit_from_dict(d; source_path=path_abs, source_file_sha256=sha)
end

function save_tgmm_h5(
    path::AbstractString,
    fit::TGMMFit;
    transformation_override=nothing,
    include_unknown=true,
    preserve_unchanged=true,
)
    path_abs = abspath(path)
    d = tgmmfit_to_dict(fit; transformation_override=transformation_override, include_unknown=include_unknown)

    if preserve_unchanged && !isnothing(fit._loaded_dict_snapshot) && !isnothing(fit._source_path)
        if fit._source_path == path_abs && _deep_equal(_canonicalize_for_compare(d), _canonicalize_for_compare(fit._loaded_dict_snapshot))
            return nothing
        end
    end

    save_dict_h5(path_abs, d)
    fit._source_path = path_abs
    fit._source_file_sha256 = _compute_sha256(path_abs)
    fit._loaded_dict_snapshot = deepcopy(d)
    nothing
end
