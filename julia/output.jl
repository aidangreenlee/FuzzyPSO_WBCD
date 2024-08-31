mutable struct output
    # struct for holding data to save into a file at the end of the run
    # including option debug boolean that may or may not be used
    debug::Bool
    output_dir::String
    number_of_runs::Int
    number_of_clusters::Int
    seeds::Array{Int,1}
    position_data::Array{Float64,3}
    fitness_vec::Array{Float64,2} # min/average/max fitness across particles
    global_best_fitness::Array{Float64,1}
    testing_vec::Array{Float64,1}
    KD_fitness_vec::Array{Float64,1}
    confusion::Array{Int,2}


    function output(number_of_runs::Int, number_of_clusters::Int, Psize::Int, debug::Bool; output_dir::String="./temp")
        position_data = Array{Float64,3}(undef, Psize, number_of_clusters, number_of_runs)
        fitness_vec = Array{Float64,2}(undef, number_of_runs, 3)
        global_best_fitness = Array{Float64}(undef, number_of_runs)
        seeds = Array{Int,1}(undef, number_of_runs)
        testing_vec = Array{Float64,1}(undef, number_of_runs)
        KD_fitness_vec = Array{Float64,1}(undef, number_of_runs)
        confusion = Array{Int,2}(undef, number_of_runs, 4)
        new(debug, output_dir, number_of_runs, number_of_clusters, seeds, position_data, fitness_vec, global_best_fitness, testing_vec, KD_fitness_vec, confusion)
    end
end