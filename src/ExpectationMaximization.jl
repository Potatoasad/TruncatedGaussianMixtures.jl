mutable struct ExpectationMaximization{T <: TruncatedMixture, L, K}
	data::L
	mix::T
	tol::Float64
	zⁿₖ::K
	score::Float64
	converged::Bool
end

mat2vecs(A) = [A[:,i] for i in 1:size(A,2)]

fixtype(x) = x;
fixtype(x::Matrix) = mat2vecs(x);
	
ExpectationMaximization(data, mix, tol, znk) = ExpectationMaximization(fixtype(data), mix, tol, znk, 0.0, false)

function ExpectationMaximization(data, n_comps; bounds1=[0.0,1.0], bounds2=[0.0,1.0],tol=1e-16)
	init = initialize(data, n_comps, bounds1, bounds2)
	data = fixtype(data)
	zⁿₖ = [Zⁿ(init, point) for point ∈ data];
	ExpectationMaximization(data, init, tol, zⁿₖ)
end


function score(EM::ExpectationMaximization)
	Y = fixtype(EM.data);
	mix = EM.mix;
	total = 0.0
	for k ∈ 1:length(mix.μ1)
		kernel1 = kernel(mix.μ1[k], mix.σ1[k], mix.bounds1)
		kernel2 = kernel(mix.μ2[k], mix.σ2[k], mix.bounds2)
		for n ∈ 1:length(Y)
			total += (EM.zⁿₖ[n][k]*(log(mix.η[k]) + loglikelihood(kernel1, Y[n][1]) + loglikelihood(kernel2, Y[n][2])))
		end
	end
	return total
end


function update!(EM::ExpectationMaximization)
	mix = EM.mix
	μ1 = deepcopy(mix.μ1); μ2 = deepcopy(mix.μ2)
	σ1 = deepcopy(mix.σ1); σ2 = deepcopy(mix.σ2)
	η = deepcopy(mix.η);

	Y = EM.data;
	prev_score = EM.score[1]
	
	bounds1 = mix.bounds1; bounds2 = mix.bounds2
	N_components = length(η)
	N_data = length(Y)

	#### E-step & M-step
	zⁿₖ = [Zⁿ(mix, point) for point ∈ Y];
	for k ∈ 1:N_components
		NewKernelR = truncated(Normal(0,σ1[k]),bounds1[1]-μ1[k], bounds1[2]-μ1[k])
		NewKernelΦ = truncated(Normal(0,σ2[k]),bounds2[1]-μ2[k], bounds2[2]-μ2[k])
		# η
		mix.η[k] = mean(zⁿₖ[n][k] for n ∈ 1:N_data)
		normalization = mix.η[k]*N_data

		# μ
		mₖ = [mean(NewKernelR), mean(NewKernelΦ)]
		μ_vec = sum(zⁿₖ[n][k]*Y[n] for n ∈ 1:N_data)/(normalization) .- mₖ
		mix.μ1[k] = μ_vec[1]; mix.μ2[k] = μ_vec[2];

		# Σ
		#Hₖ = -mₖ*mₖ'
		Σʼ = sum(zⁿₖ[n][k]*(Y[n]-μ_vec)*(Y[n]-μ_vec)' for n ∈ 1:N_data)/normalization
		Σʼ = Σʼ .- mₖ*mₖ'
		mix.σ1[k] = sqrt(Σʼ[1,1]);
		mix.σ2[k] = sqrt(Σʼ[2,2]);
	end

	#### Evaluation Check
	EM.score = score(EM)
	#@show (EM.score[1] - prev_score)
	EM.converged = ((EM.score - prev_score) ≤ EM.tol)
end
