import ClusterAnalysis
import Random
Random.seed!(1337)

include("./data.jl")
include("./cluster_data.jl")
include("./NFS.jl")
include("./PSO.jl")


#number_of_clusters = parse(Int, ARGS[1])
#number_of_particles = parse(Int, ARGS[2])
#Vₘₐₓ = parse(Float64, ARGS[3])
#W = parse(Float64, ARGS[4])
#φ₁ = parse(Float64, ARGS[5])
#φ₂ = parse(Float64, ARGS[6])
#debug = false

# Conversion
Base.convert(::Type{Matrix{Float64}}, x::Vector{T}) where {T<:Real} = reshape(Float64.(x),length(x),1)

# main function to call single instatiation of Vmax/W/φ₁/φ₂
# this can be used after training to evaluate final parameters but should not be
# used to train Optuna because it clusters the data
# can be updated later 
function main(Vmax::Float64, W::Float64, φ₁::Float64, φ₂::Float64)
    #Random.seed!(1337)
    number_of_particles = 3
    iterations = 10
    clusters, DATA = loadData(number_of_particles, iterations) # 3 clusters, 10 training set splits

    fitness = WBCD_algorithm(clusters, DATA, number_of_particles, Vmax, W, φ₁, φ₂, debug=false)

    # average fitness of all training sets
    #println("Fitness Vector:", fitness)
    return fitness
end

# load the data, only forming a single cluster/training set
function loadData(number_of_clusters::Int; filepath::String="./data/diagnostic.data")
    #Random.seed!(1337)
    DATA = WBCD_data(filepath,nvars=10)
    #clusters = ClusterAnalysis.kmeans(DATA.training_x', number_of_clusters)
    clusters = cluster_data(DATA.training_x[:,1:10], DATA.training_d, number_of_clusters)
    println("Clusters generated...")
    return clusters, DATA
end

# Cluster the data into training sets <iterations> number of times
function loadData(number_of_clusters::Int, iterations::Int; filepath::String="./data/diagnostic.data")
    Random.seed!(1337)
    cluster_vec = Vector{cluster_data}(undef, iterations)
    DATA_vec = Vector{WBCD_data}(undef, iterations)
    for i = 1:iterations
        cluster_vec[i], DATA_vec[i] = loadData(number_of_clusters, filepath = filepath)
    end
    return cluster_vec, DATA_vec
end

#function WBCD_algorithm(clusters::Vector{cluster_data}, DATA::Vector{WBCD_data}, number_of_particles::Int, Vₘₐₓ::Float64, W::Float64, φ₁::Float64, φ₂::Float64; debug::Bool=false, seed::Int=1337)
#    seed_start = seed
#    num_seeds = 10
#    fitness_vec = Matrix{Float64}(undef, length(clusters), num_seeds)
#    for i = 1:length(clusters)
#        for p = 1:num_seeds
#            Random.seed!(seed_start + p) # TODO save seed to debug
#            fitness_vec[i, p] = WBCD_algorithm(clusters[i], DATA[i], number_of_particles, Vₘₐₓ, W, φ₁, φ₂, debug=debug, seed=seed)
#        end
#    end
#    return mean(fitness_vec)
#end

function WBCD_algorithm(clusters::Array{cluster_data}, DATA::Array{WBCD_data}, number_of_particles::Int, Vₘₐₓ::Float64, W::Float64, φ₁::Float64, φ₂::Float64; debug::Bool=false, seed::Int=1337)
    Random.seed!(seed)
    PSO_ = PSO(clusters, DATA,W=W,φ₁=φ₁,φ₂=φ₂,placement="random",M=number_of_particles, Vmax=Vₘₐₓ)

    println("Running particle swarm...")

    if !isdir("outputs")
        mkdir("outputs")
    end

    if debug
        io = Vector{IOStream}(undef, PSO_.M)
        for m = 1:PSO_.M
            io[m] = open("outputs/output$(m).txt","w")
            write(io[m], "particleSize: $(size(PSO_.particles[m].position,1))\n")
            write(io[m], "W:$(PSO_.W), phi_1:$(PSO_.φ₁), phi_2:$(PSO_.φ₂)\n")
        end
    end

    fitness = Vector{Float64}(undef, PSO_.M)
    if debug
        fitness_file = open("outputs/fitness_$(seed).txt", "w")
    end
    
    mean_fitness = 0
    previous_mean_fitness = 0
    delta_fitness = 9999
    ε = 1e-3
    p = 1

    buff_size = 20
    delta_buff = Vector{Float64}(undef, 20)
    count = 1
    #while (delta_fitness > ε || mean_fitness > 0.6 || p < 100) && p < 2000
    while ( p < 100 || any(delta_buff .> ε)) && p < 2000
#        println(delta_fitness," ",mean_fitness," ",p)
        update_PSO!(PSO_, data=DATA, ThresholdMethod="normal")
        for m = 1:PSO_.M
        fitness[m] = PSO_.particles[m].H
            if debug
                writedlm(io[m], [p; PSO_.particles[m].H;vec(PSO_.particles[m].position); vec(PSO_.particles[m].velocity)]',", ")
            end
        end
        if debug
            writedlm(fitness_file, [minimum(fitness); maximum(fitness);
                                mean(fitness);    median(fitness);
                                PSO_.global_best.H]');
        end
        mean_fitness = mean(map(x->x.H, PSO_.particles))
        min_fitness  = minimum(map(x->x.H, PSO_.particles))
        max_fitness  = maximum(map(x->x.H, PSO_.particles))

        Δ = max_fitness - min_fitness
        delta_buff[count] = Δ

        count = count + 1
        if count > buff_size
            count = 1
        end

        delta_fitness = abs(mean_fitness - previous_mean_fitness)
        #println(p," ", delta_fitness)

        previous_mean_fitness = mean_fitness
        p = p + 1
    end
    println(delta_fitness," ",mean_fitness," ",p)
    println("buffer: ", delta_buff)

    final_fitness = PSO_.global_best.H
    println("Testing Set Accuracy: ", PSO_fitness(PSO_.global_best, DATA, test_train="test"))
    
    if any(delta_buff .> ε)
        tmp = delta_buff .- ε
        tmp[tmp .< 0] .= 0
        final_fitness *= 1-maximum(abs.(tmp))
    end

    println("End training: ", final_fitness)
    # close all open files
    if debug
    close(fitness_file)
    for m = 1:PSO_.M
    close(io[m])
    end
    end

    println("Done in ", p, " iterations")
    return final_fitness
end