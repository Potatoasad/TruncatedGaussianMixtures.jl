struct CovarianceBlockStructure{T, L}
	design_vector::Vector{T}
	block_dictionary::Dict{T, Vector{L}}
	block_labels::Vector{T}
	block_indices::Vector{Vector{L}}
end

function CovarianceBlockStructure(d::Vector)
	book = Dict{eltype(d),Set{Int64}}()
	for x in d
		book[x] = Set(tuple())
	end
	for i in 1:length(d)
		book[d[i]] = union(Set((i,)), book[d[i]])
	end
	book2 = Dict{eltype(d),Vector{Int64}}()
	block_labels = eltype(d)[]
	block_indices = []
	for (k,v) in book
		push!(block_labels, k)
		push!(block_indices, sort(collect(v)))
		book2[k] = sort(collect(v))
	end
	CovarianceBlockStructure{eltype(d), Int64}(d, book2, identity(block_labels), identity.(block_indices))
end

function project_onto_block!(Σ, bs::CovarianceBlockStructure)
	b = bs.design_vector
	for i in 1:length(b)
		for j in 1:length(b)
			if b[i] != b[j]
				Σ[i,j] = zero(eltype(Σ))
			end
		end
	end
end