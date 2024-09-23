using DelimitedFiles
using LinearAlgebra
using Statistics
using StatsBase

export WBCD_data



mutable struct WBCD_data
    malignant_train::Matrix{Float64}
    malignant_test::Matrix{Float64}
    benign_train::Matrix{Float64}
    benign_test::Matrix{Float64}
    names::Vector{String}

    function WBCD_data(filepath = "data/diagnostic.data"; nvars::Int64 = 30)
        data = readdlm(filepath, ',', Any, '\n')

        # get diagnosis
        d = map(x-> x == "M" ? 1 : 0, data[:,2])

        ids = data[:,1]
        malignant_ids = data[Bool.(d),1]
        benign_ids = data[.!Bool.(d),1]

        data = data[:,3:end] # strip ID and diagnosis columns from data
        data = data./maximum.(eachcol(data))'*10 # rescale between 0 and 10
        malignant_data = data[Bool.(d),:]
        benign_data = data[.!Bool.(d),:]

        training_set_size = Int(4*round(size(malignant_data,1)/5)) # 50/50 benign/malignant training set
        malignant_train_index = sample(1:size(malignant_data,1), training_set_size, replace=false)
        malignant_test_index = setdiff(1:size(malignant_data,1), malignant_train_index)
        benign_test_index = sample(1:size(benign_data,1), length(malignant_test_index), replace=false)
        benign_train_index = setdiff(1:size(benign_data,1), benign_test_index)

        malignant_train = malignant_data[malignant_train_index, 1:nvars]
        malignant_test = malignant_data[malignant_test_index, 1:nvars]
        benign_train = benign_data[benign_train_index, 1:nvars]
        benign_test = benign_data[benign_test_index, 1:nvars]
        # WBCD variable names
        names = ["radius","texture","perimeter","area","smoothness","compactness","concavity","concave_points","symmetry","fractal_dimension"]
        names = [names; map(s -> "σ_"*s, names); map(s -> "err_"*s, names)]

        # Turn data into a dictionary -- not using because it probably makes the math annoying
        # Dict{String, typeof(view(training_x,:,1))}(zip(names, eachcol(training_x)))
        
        new(malignant_train, malignant_test, benign_train, benign_test, names)
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

struct TrainData
    training_x::Matrix{Float64}
    training_d::Vector{Int}
    testing_x::Matrix{Float64}
    testing_d::Vector{Int}
    names::Vector{String}

    function TrainData(data::WBCD_data)
        train_size = size(data.malignant_train,1)
        test_size = size(data.malignant_test,1)
        benign_index = sample(1:size(data.benign_train,1),train_size,replace=false)
        training_x = [data.malignant_train; data.benign_train[benign_index,:]]
        training_d = [ones(train_size); zeros(train_size)]
        testing_x = [data.malignant_test; data.benign_test]
        testing_d = [ones(test_size); zeros(test_size)]
        
        new(training_x, training_d, testing_x, testing_d, data.names)
    end
end  