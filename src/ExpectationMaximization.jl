
mutable struct ExpectationMaximization{CVS <: AbstractCovarianceStructure, L, T, K}
	data::L
	mix::T
	tol::Float64
	zⁿₖ::K
	score::Float64
	converged::Bool
	cov_type::CVS
end

fixtype(x) = x;
fixtype(x::Matrix) = mat2vecs(x);
	
ExpectationMaximization(data, mix, tol, znk) = ExpectationMaximization(fixtype(data),mix, tol, znk, 0.0, false, covariance_type(mix)())

covariance_type(x::ExpectationMaximization{CVS}) where {CVS <: AbstractCovarianceStructure} = CVS

n_components(x::MixtureModel) = length(x.components)
n_components(x::ExpectationMaximization) = length(x.mix.components)

function Zⁿ(x::MixtureModel, datapoint::Vector)
	N_kernels = length(x.components)
	unnormalized = zeros(N_kernels);
	for k ∈ 1:N_kernels
		unnormalized[k] = log(x.prior.p[k]) + Distributions.logpdf(x.components[k], datapoint)
	end
	return exp.(unnormalized .- LogExpFunctions.logsumexp(unnormalized))
end

function ExpectationMaximization(data, n_comps; cov=:diag, a=[0.0,0.0], b=[1.0,1.0],tol=1e-16)
	init = initialize(data, n_comps, a, b; cov=cov)
	data = fixtype(data)
	zⁿₖ = [Zⁿ(init, point) for point ∈ data];
	ExpectationMaximization(data, init, tol, zⁿₖ)
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

function update!(EM::ExpectationMaximization{CVS}) where {CVS <: DiagonalCovariance}
	mix = EM.mix
	μs, Σs, η = initialize_like(mix)

	Y = EM.data;
	prev_score = EM.score
	
	a = mix.components[1].a; b = mix.components[1].b
	d = length(mix)
	N_components = length(η)
	N_data = length(Y)

	#### E-step & M-step
	EM.zⁿₖ = [Zⁿ(mix, point) for point ∈ Y];
	for k ∈ 1:N_components
		μₖ = mix.components[k].normal.μ
		Σₖ = diag(mix.components[k].normal.Σ)
		
		NewKernels = [truncated(Normal(0.0,√(Σₖ[i])),a[i]-μₖ[i], b[i]-μₖ[i]) for i ∈ 1:d]

		# η
		η[k] = mean(EM.zⁿₖ[n][k] for n ∈ 1:N_data)
		normalization = η[k]*N_data

		# μ
		mₖ = mean.(NewKernels)
		μ_vec = sum(EM.zⁿₖ[n][k]*Y[n] for n ∈ 1:N_data)/(normalization) .- mₖ

		# Σ
		Ms = var.(NewKernels) .+ mₖ.^2
		Hₖ = Σₖ .- Ms
		Σʼ = sum(EM.zⁿₖ[n][k]*(Y[n]-μ_vec)*(Y[n]-μ_vec)' for n ∈ 1:N_data)/normalization
		Σʼdiag = diag(Σʼ) .+ Hₖ
		
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


function update!(EM::ExpectationMaximization{CVS}) where {CVS <: FullCovariance}
	mix = EM.mix
	#μs = deepcopy(mix.μ1); μ2 = deepcopy(mix.μ2)
	#σ1 = deepcopy(mix.σ1); σ2 = deepcopy(mix.σ2)
	#η = deepcopy(mix.η);
	μs, Σs, η = initialize_like(mix)

	Y = EM.data;
	prev_score = EM.score
	
	a = EM.mix.components[1].a; b = EM.mix.components[1].b
	d = length(mix)
	N_components = length(η)
	N_data = length(Y)

	#### E-step & M-step
	EM.zⁿₖ = [Zⁿ(mix, point) for point ∈ Y];
	for k ∈ 1:N_components
		μₖ = mix.components[k].normal.μ
		Σₖ = mix.components[k].normal.Σ
		NewKernel = TruncatedMvNormal(MvNormal(zero(μₖ), Σₖ), a .- μₖ, b .- μₖ)

		M1,M2 = moments(NewKernel)
		
		# η
		η[k] = mean(EM.zⁿₖ[n][k] for n ∈ 1:N_data)
		normalization = η[k]*N_data

		# μ
		mₖ = M1
		μ_vec = sum(EM.zⁿₖ[n][k]*Y[n] for n ∈ 1:N_data)/(normalization) .- mₖ

		# Σ
		Hₖ = Σₖ .- M2
		Σʼ = sum(EM.zⁿₖ[n][k]*(Y[n]-μ_vec)*(Y[n]-μ_vec)' for n ∈ 1:N_data)/normalization
		Σʼ .= Σʼ .+ Hₖ
		# Impose hermiticity if lost:
		Σʼ = (Σʼ .+ Σʼ')./2
		for i ∈ 1:d
			μs[i,k] = μ_vec[i]
			for j ∈ 1:d
				Σs[i,j,k] = Σʼ[i,j]
			end
		end
	end
	EM.mix = make_mixture(μs, Σs, η, a, b)

	#### Evaluation Check
	EM.score = score(EM)
	#@show (EM.score[1] - prev_score)
	EM.converged = (abs(EM.score - prev_score) ≤ EM.tol)
end

