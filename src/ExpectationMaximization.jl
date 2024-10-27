
mutable struct ExpectationMaximization{CVS <: AbstractCovarianceStructure, L, T, K, S}
	data::L
	mix::T
	tol::Float64
	zⁿₖ::K
	score::Float64
	converged::Bool
	cov_type::CVS
	block_structure::Vector{Int64}
	weights::S
end

fixtype(x) = x;
fixtype(x::AbstractMatrix) = mat2vecs(x);
	
ExpectationMaximization(data::AbstractMatrix, args...) = ExpectationMaximization(StandardData(fixtype(data)), args...)

ExpectationMaximization(data::AbstractDataset, mix, tol, znk) = ExpectationMaximization(fixtype(data),mix, tol, znk, 0.0, false, covariance_type(mix)(), zeros(Int64, length(mix)), weight_vector(length(fixtype(data))))
ExpectationMaximization(data::AbstractDataset, mix, tol, znk, block_structure) = ExpectationMaximization(data, mix, tol, znk) # Ignore by default
ExpectationMaximization(data::AbstractDataset, mix, tol, znk, block_structure::Vector{Int64}) = ExpectationMaximization(fixtype(data),mix, tol, znk, 0.0, false, covariance_type(mix)(), block_structure, weight_vector(length(fixtype(data))))

ExpectationMaximization(data::AbstractDataset, mix, tol, znk, weights::L) where {L <: StatsBase.AbstractWeights} = ExpectationMaximization(fixtype(data),mix, tol, znk, 0.0, false, covariance_type(mix)(), zeros(Int64, length(mix)), Weights(weights ./ sum(weights)))
ExpectationMaximization(data::AbstractDataset, mix, tol, znk, block_structure, weights::L) where {L <: StatsBase.AbstractWeights} = ExpectationMaximization(data, mix, tol, znk, Weights(weights ./ sum(weights))) # Ignore by default
ExpectationMaximization(data::AbstractDataset, mix, tol, znk, block_structure::Vector{Int64}, weights::L) where {L <: StatsBase.AbstractWeights} = ExpectationMaximization(fixtype(data),mix, tol, znk, 0.0, false, covariance_type(mix)(), block_structure, Weights(weights ./ sum(weights)))

covariance_type(x::ExpectationMaximization{CVS}) where {CVS <: AbstractCovarianceStructure} = CVS

n_components(x::MixtureModel) = length(x.components)
n_components(x::ExpectationMaximization) = length(x.mix.components)

function Zⁿ(x::MixtureModel, datapoint::AbstractVector)
	N_kernels = length(x.components)
	unnormalized = zeros(N_kernels);
	for k ∈ 1:N_kernels
		unnormalized[k] = log(x.prior.p[k]) + Distributions.logpdf(x.components[k], datapoint)
	end
	return exp.(unnormalized .- LogExpFunctions.logsumexp(unnormalized))
end

function Zⁿ(x::MixtureModel, datapoint::AbstractVector, β::Float64)
	N_kernels = length(x.components)
	unnormalized = zeros(N_kernels);
	for k ∈ 1:N_kernels
		unnormalized[k] = log(x.prior.p[k]) + β * Distributions.logpdf(x.components[k], datapoint)
	end
	return exp.(unnormalized .- LogExpFunctions.logsumexp(unnormalized))
end

function Zⁿ!(unnormalized, x::MixtureModel, datapoint::AbstractVector)
	N_kernels = length(x.components)
	for k ∈ 1:N_kernels
		unnormalized[k] = log(x.prior.p[k]) + Distributions.logpdf(x.components[k], datapoint)
	end
	log_norm = LogExpFunctions.logsumexp(unnormalized)
	for k ∈ 1:N_kernels
		unnormalized[k] = exp(unnormalized[k] - log_norm)
	end
end

function Zⁿ!(unnormalized, x::MixtureModel, datapoint::AbstractVector, β::Float64)
	N_kernels = length(x.components)
	for k ∈ 1:N_kernels
		unnormalized[k] = log(x.prior.p[k]) + β * Distributions.logpdf(x.components[k], datapoint)
	end
	log_norm = LogExpFunctions.logsumexp(unnormalized)
	for k ∈ 1:N_kernels
		unnormalized[k] = exp(unnormalized[k] - log_norm)
	end
end

function ExpectationMaximization(data, n_comps; cov=:diag, a=[0.0,0.0], b=[1.0,1.0], tol=1e-16, block_structure=false, β=1.0, weights=nothing, unbiasing=:default)
	if (unbiasing == :default) | (unbiasing isa DefaultUnbiasing)
		data = StandardData(data)
	elseif (unbiasing == :boundary) | (unbiasing isa BoundaryUnbiasing)
		if unbiasing == :boundary
			unbiasing = BoundaryUnbiasing(ones(Bool, length(a)))
		end
		data = fixtype(data)
		data = BoundaryUnbiasedData(data, a , b, unbiasing);
	else
		throw(NotImplementedError("Can't identify unbiasing type $(unbiasing)"))
	end
	weights = compute_weights(weights, length(fixtype(data)))
	#print(weights)
	init = initialize(data, n_comps, a, b; cov=cov, weights=weights)
	data = fixtype(data)
	#@show typeof(get_data(data))
	zⁿₖ = [Zⁿ(init, point) for point ∈ data];
	ExpectationMaximization(data, init, tol, zⁿₖ, block_structure, weights)
end

function score(EM::ExpectationMaximization)
	Y = fixtype(EM.data);
	mix = EM.mix;
	total = 0.0
	for k ∈ 1:n_components(EM)
		for n ∈ 1:length(Y)
			total += (EM.zⁿₖ[n][k]*(log(mix.prior.p[k]) + Distributions.logpdf(mix.components[k], Y[n])))
		end
	end
	return total/(n_components(EM)*length(Y))
end

#"""
#Computes the normalized entropy as shown in https://www.cs.cmu.edu/~rsalakhu/papers/emecg.pdf
#Used to decide whether to do EM or ECG
#"""
function normalized_entropy(zⁿₖ)
	N = length(zⁿₖ); K = length(zⁿₖ[1]);
	return (-1/(N*log(K)))sum(zⁿₖ[n][k] * log(zⁿₖ[n][k]) for k in 1:K, n in 1:N)
end

function update!(EM::ExpectationMaximization{CVS}, β::Float64) where {CVS <: DiagonalCovariance}
	mix = EM.mix
	μs, Σs, η = initialize_like(mix)

	Y = get_data(EM.data);
	prev_score = EM.score
	
	a = mix.components[1].a; b = mix.components[1].b
	d = length(mix)
	N_components = length(η)
	N_data = length(Y)

	μ_vec = zeros(d);
	Σʼdiag = zeros(d);

	#### E-step & M-step
	#EM.zⁿₖ = [Zⁿ(mix, point, β) for point ∈ Y];
	for n ∈ 1:length(Y)
		Zⁿ!(EM.zⁿₖ[n], mix, Y[n], β)
	end
	W = EM.weights .* N_data

	Hₜ = normalized_entropy(EM.zⁿₖ)

	if Hₜ > 0.5
		### Conjugate Gradient
		#@show a, b

		for k ∈ 1:N_components
			μₖ = mix.components[k].normal.μ
			Σₖ = diag(mix.components[k].normal.Σ)
			
			#NewKernels = [truncated(Normal(0.0,√(Σₖ[i])),a[i]-μₖ[i], b[i]-μₖ[i]) for i ∈ 1:d]

			# η
			η[k] = mean(EM.zⁿₖ[n][k] * W[n] for n ∈ 1:N_data)
			normalization = η[k]*N_data

			# μ
			#mₖ = mean.(NewKernels)

			#weighted_mean = sum(EM.zⁿₖ[n][k]*Y[n]* W[n] for n ∈ 1:N_data)/(normalization)
			#weighted_covariance = diag(sum(EM.zⁿₖ[n][k]* W[n]*(Y[n]-weighted_mean)*(Y[n]-weighted_mean)' for n ∈ 1:N_data)/normalization)

			weighted_mean, weighted_covariance = mean_and_cov(EM.data,EM.zⁿₖ, W, k)

			for i ∈ 1:d
				μ_vec[i], Σʼdiag[i] = iterate_to_conclusion(weighted_mean[i], weighted_covariance[i,i], a[i], b[i])
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
		end
	else
		### Expectation Maximization

		for k ∈ 1:N_components
			μₖ = mix.components[k].normal.μ
			Σₖ = diag(mix.components[k].normal.Σ)

			#@show a, b, μₖ, Σₖ
			
			NewKernels = [truncated(Normal(0.0,√(Σₖ[i])),a[i]-μₖ[i], b[i]-μₖ[i]) for i ∈ 1:d]

			# η
			η[k] = mean(EM.zⁿₖ[n][k] * W[n] for n ∈ 1:N_data)
			normalization = η[k]*N_data

			weighted_mean, weighted_covariance = mean_and_cov(EM.data,EM.zⁿₖ, W, k)
			#@show weighted_mean, diag(weighted_covariance), var.(NewKernels)

			# μ
			mₖ = mean.(NewKernels)
			μ_vec = weighted_mean .- mₖ

			# Σ
			#Ms = var.(NewKernels) .+ mₖ.^2
			#Hₖ = Σₖ .- Ms
			#Σʼ = sum(EM.zⁿₖ[n][k]* W[n]*(Y[n]-μ_vec)*(Y[n]-μ_vec)' for n ∈ 1:N_data)/normalization
			#Σʼdiag = diag(Σʼ) .+ Hₖ
			Σʼdiag = diag(weighted_covariance) .+ Σₖ .- var.(NewKernels)

			# If we find a particular kernel with only one point, we set it's weight to zero
			if any( Σʼdiag .≤ zero(eltype(Σʼdiag)) ) 
				@show Σʼdiag
				η[k] = zero(η[k])
				Σʼdiag .+= (((1/N_data) .* (b .- a)).^2) .* ones(eltype(Σʼdiag), length(Σʼdiag))
			end
			
			for i ∈ 1:d
				μs[i,k] = μ_vec[i]
				Σs[i,k] = Σʼdiag[i]
				#Σs[i,k] = Σʼ[i,i]
			end
		end
	end

	#@show μs, Σs
	#println(η)

	EM.mix = make_mixture(μs, Σs, η, a, b)

	#### Evaluation Check
	#@show EM.converged
	EM.score = score(EM)
	#@show (EM.score[1] - prev_score)
	EM.converged = (abs(EM.score - prev_score) ≤ EM.tol)
end

update!(EM::ExpectationMaximization{CVS}) where {CVS <: DiagonalCovariance} = update!(EM, 1.0)



function update_old!(EM::ExpectationMaximization{CVS}, β::Float64) where {CVS <: DiagonalCovariance}
	mix = EM.mix
	μs, Σs, η = initialize_like(mix)

	Y = get_data(EM.data);
	prev_score = EM.score
	
	a = mix.components[1].a; b = mix.components[1].b
	d = length(mix)
	N_components = length(η)
	N_data = length(Y)

	#### E-step & M-step
	#EM.zⁿₖ = [Zⁿ(mix, point, β) for point ∈ Y];
	for n ∈ 1:length(Y)
		Zⁿ!(EM.zⁿₖ[n], mix, Y[n], β)
	end
	W = EM.weights .* N_data
	for k ∈ 1:N_components
		μₖ = mix.components[k].normal.μ
		Σₖ = diag(mix.components[k].normal.Σ)
		
		NewKernels = [truncated(Normal(0.0,√(Σₖ[i])),a[i]-μₖ[i], b[i]-μₖ[i]) for i ∈ 1:d]

		# η
		η[k] = mean(EM.zⁿₖ[n][k] * W[n] for n ∈ 1:N_data)
		normalization = η[k]*N_data

		# μ
		mₖ = mean.(NewKernels)
		μ_vec = sum(EM.zⁿₖ[n][k]*Y[n]* W[n] for n ∈ 1:N_data)/(normalization) .- mₖ

		# Σ
		Ms = var.(NewKernels) .+ mₖ.^2
		Hₖ = Σₖ .- Ms
		Σʼ = sum(EM.zⁿₖ[n][k]* W[n]*(Y[n]-μ_vec)*(Y[n]-μ_vec)' for n ∈ 1:N_data)/normalization
		Σʼdiag = diag(Σʼ) .+ Hₖ

		# If we find a particular kernel with only one point, we set it's weight to zero
		if any( Σʼdiag .≤ zero(eltype(Σʼdiag)) ) 
			η[k] = zero(η[k])
			Σʼdiag .+= (((1/N_data) .* (b .- a)).^2) .* ones(eltype(Σʼdiag), length(Σʼdiag))
		end
		
		for i ∈ 1:d
			μs[i,k] = μ_vec[i]
			Σs[i,k] = Σʼdiag[i]
			#Σs[i,k] = Σʼ[i,i]
		end
	end
	#println(η)

	EM.mix = make_mixture(μs, Σs, η, a, b)

	#### Evaluation Check
	EM.score = score(EM)
	#@show (EM.score[1] - prev_score)
	EM.converged = (abs(EM.score - prev_score) ≤ EM.tol)
end

update_old!(EM::ExpectationMaximization{CVS}) where {CVS <: DiagonalCovariance} = update!(EM, 1.0)


function update_old!(EM::ExpectationMaximization{CVS}, β::Float64) where {CVS <: FullCovariance}
	mix = EM.mix
	#μs = deepcopy(mix.μ1); μ2 = deepcopy(mix.μ2)
	#σ1 = deepcopy(mix.σ1); σ2 = deepcopy(mix.σ2)
	#η = deepcopy(mix.η);
	μs, Σs, η = initialize_like(mix)

	Y = get_data(EM.data);
	prev_score = EM.score
	
	a = EM.mix.components[1].a; b = EM.mix.components[1].b
	d = length(mix)
	N_components = length(η)
	N_data = length(Y)

	#### E-step & M-step
	EM.zⁿₖ = [Zⁿ(mix, point, β) for point ∈ Y];
	W = EM.weights.* N_data
	for k ∈ 1:N_components
		μₖ = mix.components[k].normal.μ
		Σₖ = mix.components[k].normal.Σ
		NewKernel = TruncatedMvNormal(MvNormal(zero(μₖ), Σₖ), a .- μₖ, b .- μₖ)

		M1,M2 = moments(NewKernel)
		
		# η
		η[k] = mean(EM.zⁿₖ[n][k]* W[n] for n ∈ 1:N_data)
		normalization = η[k]*N_data

		# μ
		mₖ = M1
		μ_vec = sum(EM.zⁿₖ[n][k]*Y[n]* W[n] for n ∈ 1:N_data)/(normalization) .- mₖ


		# Σ
		Hₖ = Σₖ .- M2
		Σʼ = sum(EM.zⁿₖ[n][k]* W[n]*(Y[n]-μ_vec)*(Y[n]-μ_vec)' for n ∈ 1:N_data)/normalization
		Σʼ .= Σʼ .+ Hₖ
		# Impose hermiticity if lost:
		Σʼ = (Σʼ .+ Σʼ')./2

		for i ∈ 1:d
			μs[i,k] = μ_vec[i]
			for j ∈ 1:d
				# Remove correlations that should not exist
				if EM.block_structure[i] == EM.block_structure[j]
					Σs[i,j,k] = Σʼ[i,j]
				else
					Σs[i,j,k] = zero(Σʼ[i,j])
				end
			end
		end
	end
	EM.mix = make_mixture(μs, Σs, η, a, b)

	#### Evaluation Check
	EM.score = score(EM)
	#@show (EM.score[1] - prev_score)
	EM.converged = (abs(EM.score - prev_score) ≤ EM.tol)
end

update_old!(EM::ExpectationMaximization{CVS}) where {CVS <: FullCovariance} = update_old!(EM, 1.0)




function update!(EM::ExpectationMaximization{CVS}, β::Float64) where {CVS <: FullCovariance}
	mix = EM.mix
	bs = CovarianceBlockStructure(EM.block_structure)
	#μs = deepcopy(mix.μ1); μ2 = deepcopy(mix.μ2)
	#σ1 = deepcopy(mix.σ1); σ2 = deepcopy(mix.σ2)
	#η = deepcopy(mix.η);
	μs, Σs, η = initialize_like(mix)

	Y = get_data(EM.data);
	prev_score = EM.score
	
	a = EM.mix.components[1].a; b = EM.mix.components[1].b
	d = length(mix)
	N_components = length(η)
	N_data = length(Y)

	#### E-step & M-step
	for n ∈ 1:length(Y)
		Zⁿ!(EM.zⁿₖ[n], mix, Y[n], β)
	end
	#EM.zⁿₖ = [Zⁿ(mix, point, β) for point ∈ Y];
	W = EM.weights.* N_data

	M1 = zeros(length(a))
	M2 = zeros(length(a), length(a))
	throwaway = zeros(length(a))

	for k ∈ 1:N_components
		μₖ = mix.components[k].normal.μ
		Σₖ = mix.components[k].normal.Σ
		#NewKernel = TruncatedMvNormal(MvNormal(zero(μₖ), Σₖ), a .- μₖ, b .- μₖ)

		#M1,M2 = moments(NewKernel)
		EXᵢXⱼ!(M1, M2, throwaway, Σₖ, a .- μₖ, b .- μₖ, bs)
		
		# η
		η[k] = mean(EM.zⁿₖ[n][k]* W[n] for n ∈ 1:N_data)
		normalization = η[k]*N_data

		weighted_mean, weighted_covariance = mean_and_cov(EM.data,EM.zⁿₖ, W, k)

		# μ
		mₖ = M1
		#μ_vec = sum(EM.zⁿₖ[n][k]*Y[n]* W[n] for n ∈ 1:N_data)/(normalization) .- mₖ
		μ_vec = weighted_mean .- mₖ


		# Σ
		Hₖ = Σₖ .- M2
		#Σʼ = sum(EM.zⁿₖ[n][k]* W[n]*(Y[n]-μ_vec)*(Y[n]-μ_vec)' for n ∈ 1:N_data)/normalization
		Σʼ = weighted_covariance .+ (mₖ * mₖ')
		Σʼ .= Σʼ .+ Hₖ
		# Impose hermiticity if lost:
		Σʼ = (Σʼ .+ Σʼ')./2

		for i ∈ 1:d
			μs[i,k] = μ_vec[i]
			for j ∈ 1:d
				# Remove correlations that should not exist
				if EM.block_structure[i] == EM.block_structure[j]
					Σs[i,j,k] = Σʼ[i,j]
				else
					Σs[i,j,k] = zero(Σʼ[i,j])
				end
			end
		end
	end
	EM.mix = make_mixture(μs, Σs, η, a, b)

	#### Evaluation Check
	EM.score = score(EM)
	#@show (EM.score[1] - prev_score)
	EM.converged = (abs(EM.score - prev_score) ≤ EM.tol)
end

update!(EM::ExpectationMaximization{CVS}) where {CVS <: FullCovariance} = update!(EM, 1.0)

