function fit_kde(df_in::DataFrame, a, b, cols, bs::BoundaryUnbiasing; subsampling=nothing, full_subsampling=nothing, weights=nothing, fix=true, progress=true, bandwidth_scale=nothing)
    N_data = nrow(df_in)

    if full_subsampling == nothing
        df = df_in
        full_indices = collect(1:N_data)
    else
        full_subsampling = minimum([N_data, full_subsampling])
        full_indices = rand(1:N_data, full_subsampling)
        df = df_in[full_indices, :]
        N_data = nrow(df)
    end

    #subsampling = (subsampling == nothing) ? N_data :  subsampling
    if subsampling == nothing
        indices = 1:N_data
    else
        indices = rand(1:N_data, subsampling);
    end
    W = (weights == nothing) ? weight_vector(N_data) : weights

   
    

    X = fixtype(Matrix(df[indices, cols])')
    data = BoundaryUnbiasedData(X, a, b, bs; bandwidth_scale=bandwidth_scale)

    N = size(data.μ,1)
    init_kde = MixtureModel([TruncatedMvNormal(MvNormal(data.μ[i,:], data.bandwidth),a,b) for i in 1:N],ones(N)./N)

    if !fix
        return init_kde, full_indices[indices]
    end

    X_full = fixtype(Matrix(df[!, cols])')
    data_full = BoundaryUnbiasedData(X_full, a, b, bs)

    d = length(a);

    η = zeros(N)
    μs = zeros(d, N)
    Σs = zeros(d, N)
    zⁿₖ = zeros(N_data, N)

    if progress
        progressbar = Progress(N_data)
    end

    for n in 1:N_data
        Zⁿ!(@view(zⁿₖ[n,:]), init_kde, X_full[n])
        if progress
            next!(progressbar)
        end
    end

    if progress
        finish!(progressbar)
    end

    μ_vec, Σʼdiag = zeros(d), zeros(d);

    if progress
        progressbar = Progress(N)
    end


    for k ∈ 1:N
        #μₖ = init_kde.components[k].normal.μ
        #Σₖ = diag(init_kde.components[k].normal.Σ)

        #NewKernels = [truncated(Normal(0.0,√(Σₖ[i])),a[i]-μₖ[i], b[i]-μₖ[i]) for i ∈ 1:d]

        # η
        η[k] = mean(zⁿₖ[n,k] * W[n] for n ∈ 1:N_data)
        normalization = η[k]*N_data

        # μ
        #mₖ = mean.(NewKernels)

        #weighted_mean = sum(EM.zⁿₖ[n][k]*Y[n]* W[n] for n ∈ 1:N_data)/(normalization)
        #weighted_covariance = diag(sum(EM.zⁿₖ[n][k]* W[n]*(Y[n]-weighted_mean)*(Y[n]-weighted_mean)' for n ∈ 1:N_data)/normalization)

        weighted_mean, weighted_covariance = mean_and_cov(data_full,zⁿₖ, W, k)
        #@show  weighted_mean, weighted_covariance

        for i ∈ 1:d
            if weighted_covariance != zero(weighted_covariance)
                μ_vec[i], Σʼdiag[i] = iterate_to_conclusion(weighted_mean[i], weighted_covariance[i,i], a[i], b[i])
            else
                μ_vec[i], Σʼdiag[i] = weighted_mean[i], data.bandwidth[i]^2
            end
        end

        # If we find a particular kernel with only one point, we set it's weight to zero
        for i ∈ 1:d
            if Σʼdiag[i] .≤ zero(eltype(Σʼdiag))
                η[k] = zero(η[k])
                Σʼdiag .+= (((1/N_data) .* (b .- a)).^2) .* ones(eltype(Σʼdiag), length(Σʼdiag))
            end
        end
        
        for i ∈ 1:d
            μs[i,k] = μ_vec[i]
            Σs[i,k] = Σʼdiag[i]
            #Σs[i,k] = Σʼ[i,i]
        end

        if progress
            next!(progressbar)
        end
    end

    if progress
        finish!(progressbar)
    end


    make_mixture(μs, Σs, η, a, b), full_indices[indices] 
end



function LSCV(thekde, datapoints::AbstractVector)
    N = length(datapoints)

    mean(pdf(thekde, point) for point in datapoints)




end





















