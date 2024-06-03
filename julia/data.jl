using DelimitedFiles
using LinearAlgebra
using Statistics

export WBCD_data

struct WBCD_data
    training_x::Matrix{Float64}
    training_d::Vector{Int}
    testing_x::Matrix{Float64}
    testing_d::Vector{Int}
    names::Vector{String}

    function WBCD_data(filepath = "data/diagnostic.data"; nvars::Int64 = 30)
        data = readdlm(filepath, ',', Any, '\n')

        # get diagnosis
        d = map(x-> x == "M" ? 1 : 0, data[:,2])
        data = data[:,3:end] # strip ID and diagnosis columns from data
        data = data./maximum.(eachcol(data))'*10 # rescale between 0 and 10

        training_set_size = Int(2*round(size(data,1)/3))

        training_set_index = rand(1:size(data,1), training_set_size)
        testing_set_index  = setdiff(1:size(data,1), training_set_index)
       
        training_x = data[training_set_index, 1:nvars]
        training_d = d[training_set_index]
        testing_x  = data[testing_set_index, 1:nvars]
        testing_d  = d[testing_set_index]

        # WBCD variable names
        names = ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave_points","symmetry","fractal_dimension"]
        names = [names; map(s -> "σ_"*s, names); map(s -> "err_"*s, names)]

        # Turn data into a dictionary -- not using because it probably makes the math annoying
        # Dict{String, typeof(view(training_x,:,1))}(zip(names, eachcol(training_x)))
        
        new(training_x, training_d, testing_x, testing_d, names)
    end
end    

       
       
       
       