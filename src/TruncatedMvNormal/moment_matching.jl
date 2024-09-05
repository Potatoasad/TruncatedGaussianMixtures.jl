function unflatten_view_diag(X, d)
	μ, Σ = (@view X[1:d]), (@view X[(d+1):end])
end

function stopping_criteria(μΣ)
	### Please put in a criteria where if μ, Σ is inside
	### then win
	N = length(μΣ)÷2
	μ = μΣ[1:N]; Σ = μΣ[(N+1):end]
	
	if any(Σ .> 100) | any(Σ .< 1e-9)
		return true
	end
	return false
end

function is_beyond_lim(μ, σ, a, b; lower_lim=-2, upper_lim=2)
	if μ < a
		return ((a-μ)/σ) > -lower_lim
	elseif μ > b
		return ((μ-b)/σ) > upper_lim
	else
		return false
	end
end

function stopping_criteria(μΣ, a, b; lower_lim=-4, upper_lim=4)
	### Please put in a criteria where if μ, Σ is inside
	### then win
	N = length(μΣ)÷2
	μ = μΣ[1:N]; Σ = μΣ[(N+1):end]
	beyond = any(is_beyond_lim(μᵢ, sqrt(Σᵢ), aᵢ, bᵢ; lower_lim=lower_lim, upper_lim=upper_lim) for (μᵢ,Σᵢ,aᵢ,bᵢ) in zip(μ,Σ,a,b))
	if any(Σ .> 100) | any(Σ .< 1e-9) | beyond
		return true
	end
	return false
end


function fix_kernels(mu, sigma, a, b; lower_lim=-2, upper_lim=2, n=10)
    alpha = (mu - a) / sigma
    beta = (mu - b) / sigma
    too_far_left = (alpha < lower_lim) && (beta < 0)
    too_far_right = (beta > upper_lim) && (alpha > 0)

    if too_far_left || too_far_right
        sigma_fixed = sigma / sqrt(n)
    else
        sigma_fixed = sigma
    end

    if too_far_left
        mu_fixed = a - (a - mu) / n
    elseif too_far_right
        mu_fixed = b + (mu - b) / n
    else
        mu_fixed = mu
    end

    return mu_fixed, sigma_fixed
end

function fix_kernels_until_good(mu, sigma, a, b; lower_lim=-2, upper_lim=2)
    if is_beyond_lim(mu, sigma, a, b; lower_lim=lower_lim, upper_lim=upper_lim)
        for N in [2,3,4,6,8,10,14,18]
            mu_, sigma_ = fix_kernels(mu, sigma, a, b; lower_lim=lower_lim, upper_lim=upper_lim, n=N)
            if !is_beyond_lim(mu_, sigma_, a, b; lower_lim=lower_lim, upper_lim=upper_lim)
               return mu_, sigma_ 
            end
            if N==18
                #println("Failed to fix")
                #@show mu, sigma, mu_, sigma_, a, b
                return fix_kernels(mu_, sigma_, a, b; lower_lim=lower_lim, upper_lim=upper_lim, n=30)
            end
        end
    else
        return mu, sigma
    end
end

function get_mean_var_error(μΣ, p)
    N = length(μΣ) ÷ 2
    μ = μΣ[1:N]; Σ = μΣ[(N+1):end]
    target_means, target_vars, a, b = p
    dists = [truncated(Normal(μ[i], sqrt(Σ[i])), a[i], b[i]) for i in 1:length(a)]
    vcat((@. target_means - mean(dists)), (@. target_vars - var(dists)))
end

function do_iteration!(x, jac, params)
    target_means, target_vars, a, b = params
    oldx = copy(x)
    for i in 1:20
        if stopping_criteria(x) | stopping_criteria(x, a, b)
            #print("Went Past")
            break
        end
        ForwardDiff.jacobian!(jac, x -> get_mean_var_error(x, params), x)
        A = jac
        bss = -get_mean_var_error(x, params)
        y = A \ bss
        x .= y + x  # Update x in place
    end
    if stopping_criteria(x) | stopping_criteria(x, a, b)
        d = length(x)÷2
        μ,Σ = unflatten_view_diag(x, d)
        for i in 1:d
            μ1,σ1 = fix_kernels_until_good(μ[i], √(Σ[i]), a[i], b[i]; lower_lim=-5, upper_lim=5)
            x[i] = μ1
            x[i+d] = σ1^2
        end
    end
    #println(x)
end

function iterate_to_conclusion(target_means, target_vars, a, b)
    μΣ_init = vcat(target_means, target_vars)
    params = (target_means, target_vars, a, b)
    jac = zeros(length(μΣ_init), length(μΣ_init))
    #@show μΣ_init
    x = copy(μΣ_init)
    do_iteration!(x, jac, params)
    x
end

function fit_best_truncated_normal_single(df,a,b,bws)
	#μ,Σ = zeros(length(columns)), zeros(length(columns))
	#dists =[truncated(Normal(zero(μs[i]), sqrt(Σs[i])), a[i]-μs[i], b[i]-μs[i]) for i in 1:length(a)]
	#target_means = [mean(df[!, col]) - mean(dist) for (col,dist) in zip(names(df), dists)]
	#target_vars = [var(df[!, col]) - var(dist) + dist.untruncated.σ^2 for (col,dist) in zip(names(df),dists)]
    cols = names(df)
	target_means = [mean(df[!, col]) for (col) in names(df)]
	target_vars = [var(df[!, col]) for (col) in names(df)]
    #@show nrow(df)
    if nrow(df) == 1
        (μ, Σ) = (collect(df[1,cols]), bws.^2)
        return μ, Σ
    elseif nrow(df) == 0
        (μ, Σ) = ((a .+ b)./2, bws.^2)
        return μ, Σ
    end
    for i in 1:length(target_vars)
       if target_vars[i] < 0.5*bws[i]^2
           target_vars[i] = 0.5*bws[i]^2
       end
    end
    #@show target_means, target_vars
	xfinal = iterate_to_conclusion(target_means, target_vars, a, b)
	N = length(xfinal)÷2
	μ = xfinal[1:N]; Σ = xfinal[(N+1):end]
	μ, Σ
end