using DelimitedFiles
using LinearAlgebra
using Statistics
using StatsBase

export WBCD_data

mutable struct WBCD_data
    training_x::Matrix{Float64}
    training_d::Vector{Int}
    testing_x::Matrix{Float64}
    testing_d::Vector{Int}
    names::Vector{String}
    training_ids::Vector{Int}
    testing_ids::Vector{Int}

    function WBCD_data(filepath = "data/diagnostic.data"; nvars::Int64 = 30)
        data = readdlm(filepath, ',', Any, '\n')

        # get diagnosis
        d = map(x-> x == "M" ? 1 : 0, data[:,2])
        ids = data[:,1]
        data = data[:,3:end] # strip ID and diagnosis columns from data
        data = data./maximum.(eachcol(data))'*10 # rescale between 0 and 10

        training_set_size = Int(4*round(size(data,1)/5))

        # training_set_index = rand(1:size(data,1), training_set_size)
        training_set_index = sample(1:size(data,1), training_set_size, replace=false)
        testing_set_index  = setdiff(1:size(data,1), training_set_index)
       
        training_x = data[training_set_index, 1:nvars]
        training_d = d[training_set_index]
        testing_x  = data[testing_set_index, 1:nvars]
        testing_d  = d[testing_set_index]
        training_ids = ids[training_set_index]
        testing_ids = ids[testing_set_index]

        # WBCD variable names
        names = ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave_points","symmetry","fractal_dimension"]
        names = [names; map(s -> "σ_"*s, names); map(s -> "err_"*s, names)]

        # Turn data into a dictionary -- not using because it probably makes the math annoying
        # Dict{String, typeof(view(training_x,:,1))}(zip(names, eachcol(training_x)))
        
        new(training_x, training_d, testing_x, testing_d, names, training_ids, testing_ids)
    end

    function WBCD_data(original::Bool; filepath = "data/original.data")
        data = readdlm(filepath,',',Any,'\n')
        uix = unique(i -> data[i,2:end], 1:size(data,1)) # remove all repeated rows
        data = data[uix,:]
        data = data[findall(.!any.(eachrow(isnan.(data)))),:] # remove all NaN
        d = map(x-> x == 2 ? 0 : 1, data[:,11]) # 2 is benign, 4 is malignant
        ids = data[:,1]
        data = data[:,2:end-1] # strip id and diagnosis
        training_set_size = Int(4*round(size(data,1)/5))

        training_set_index = sample(1:size(data,1), training_set_size, replace=false)
        testing_set_index = setdiff(1:size(data,1), training_set_index)

        training_x = data[training_set_index, 1:9]
        training_d = d[training_set_index]
        testing_x  = data[testing_set_index, 1:9]
        testing_d  = d[testing_set_index]
        training_ids = ids[training_set_index]
        testing_ids = ids[testing_set_index]

        # WBCD variable names
        names = ["clump_thickness","cellsize_uniformity","cellshape_uniformity","marginal_adhesion","epithelial_cellsize","bare_nuclei","bland_chromatin","normal_nucleoli","mitoses"]
        #names = [names; map(s -> "σ_"*s, names); map(s -> "err_"*s, names)]

        # Turn data into a dictionary -- not using because it probably makes the math annoying
        # Dict{String, typeof(view(training_x,:,1))}(zip(names, eachcol(training_x)))
        
        new(training_x, training_d, testing_x, testing_d, names, training_ids, testing_ids)
    end
end    

       
       
       
       