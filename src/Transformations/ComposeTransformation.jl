function compose(T1::AbstractTransformation, T2::AbstractTransformation)
	Transformation(T1.domain_columns,
    (x...) -> T2.forward(T1.forward(x...)...),
    T2.image_columns,
    (x...) -> T1.inverse(T2.inverse(x...)...),
    T1.ignore_columns
	)
end