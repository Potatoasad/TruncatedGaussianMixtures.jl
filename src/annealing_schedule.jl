import Base

abstract type AbstractSchedule end

Base.iterate(S::AbstractSchedule) = Base.iterate(iterator(S))
Base.iterate(S::AbstractSchedule, state) = Base.iterate(iterator(S), state)

struct AnnealingSchedule{T} <: AbstractSchedule
	β_list::T
end

function AnnealingSchedule(;β_max=1.5, dβ_rise = 0.02, dβ_relax=0.01, N_high=50, N_post=100)
	β_schedule = vcat( 0:dβ_rise:β_max, β_max.*ones(N_high),β_max:(-dβ_relax):1.0, ones(N_post) )
	return AnnealingSchedule(β_schedule)
end

iterator(S::AnnealingSchedule) = S.β_list



