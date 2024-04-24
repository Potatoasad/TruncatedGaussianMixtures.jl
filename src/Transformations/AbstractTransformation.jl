abstract type AbstractTransformation end

forward(Tr::AbstractTransformation) = Tr.forward
domain_columns(Tr::AbstractTransformation) = Tr.domain_columns
inverse(Tr::AbstractTransformation) = Tr.inverse
image_columns(Tr::AbstractTransformation) = Tr.image_columns

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


struct Transformation{A,B,C,D,L} <: AbstractTransformation
	domain_columns::A
	forward::B
	image_columns::C
	inverse::D
	ignore_columns::L
end

Transformation(domain_columns, forward, image_columns, inverse; ignore_columns=[]) = Transformation(domain_columns, forward, image_columns, inverse, ignore_columns)

#to_chirp_mass(q) = ((q^3)/(1+q))^(1/5)
#to_chirp_mass(m1,q) = m1*f(q)
 

