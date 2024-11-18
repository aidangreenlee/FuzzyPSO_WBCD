import ClusterAnalysis
import Random
Random.seed!(1337)

include("./data.jl")
include("./cluster_data.jl")
include("./NFS.jl")
include("./PSO.jl")
include("./output.jl")

#number_of_clusters = parse(Int, ARGS[1])
#number_of_particles = parse(Int, ARGS[2])
#Vₘₐₓ = parse(Float64, ARGS[3])
#W = parse(Float64, ARGS[4])
#φ₁ = parse(Float64, ARGS[5])
#φ₂ = parse(Float64, ARGS[6])
#debug = false

function main(number_of_clusters::Int, Vmax::Float64, W::Float64, φ₁::Float64, φ₂::Float64)
    Random.seed!(1337)
    number_of_particles = 20
    #number_of_clusters = 3
    iterations = 1
    #iterations = 2
    clusters, DATA = loadData(number_of_clusters, iterations) # 3 clusters, 10 training set splits

    output_data = output(100, number_of_clusters, 21, output_dir="./temp", true)
    #for i = 1:iterations
    fitness = WBCD_algorithm(clusters, DATA, number_of_particles, Vmax, W, φ₁, φ₂, debug=output_data, iteration=1)
    #end

    # average fitness of all training sets
    #println("Fitness Vector:", fitness)
    return fitness
end

function loadData(number_of_clusters::Int; filepath::String="./data/diagnostic.data")
    #Random.seed!(1337)
    DATA = WBCD_data(filepath, nvars=10)
    TDAT = TrainData(DATA)
    #DATA = WBCD_data(true)
    #clusters = ClusterAnalysis.kmeans(DATA.training_x', number_of_clusters)
    clusters = cluster_data(TDAT.training_x, TDAT.training_d, number_of_clusters)
    println("Clusters generated...")
    return clusters, DATA
end

# Cluster the data into training sets <iterations> number of times to split training set
function loadData(number_of_clusters::Int, iterations::Int; filepath::String="./data/diagnostic.data")
    Random.seed!(1337)
    cluster_vec = Vector{cluster_data}(undef, iterations)
    DATA_vec = Vector{WBCD_data}(undef, iterations)
    for i = 1:iterations
        cluster_vec[i], DATA_vec[i] = loadData(number_of_clusters, filepath=filepath)
    end
    return cluster_vec, DATA_vec
end

function WBCD_algorithm(clusters::Vector{cluster_data}, DATA::Vector{WBCD_data}, number_of_particles::Int, Vₘₐₓ::Float64, W::Float64, φ₁::Float64, φ₂::Float64; alpha::Float64=0.2, beta::Float64=0.4, gamma::Float64=0.4, debug::output, seed::Int=1337, iteration::Int=0)
    seed_start = seed
    num_seeds = 100
    #num_seeds = 2
    KD_fitness = Array{Float64,1}(undef, length(clusters) * num_seeds)
    KD = Array{Float64,1}(undef,length(clusters) * num_seeds)
    iterations = Array{Int,1}(undef,length(clusters) * num_seeds)
    count = 1

    debug.output_dir = "outputs_5050pppp_$(alpha)_$(beta)_$(gamma)_$(clusters[1].K)"

    for i = 1:length(clusters)
        for p = 1:num_seeds
            Random.seed!(seed_start + p) # TODO save seed to debug
            println("Training Data Split!")
            TDAT = TrainData(DATA[i])
            KD_fitness[count], KD[count] , test, confusion, PSO_, iterations[count] = WBCD_algorithm(clusters[i], TDAT, number_of_particles, Vₘₐₓ, W, φ₁, φ₂, alpha=alpha, beta=beta, gamma=gamma, debug=debug, seed=seed_start + p)
            all_p_fitness = map(x->x.H, PSO_.particles)

            debug.fitness_vec[count,:] = [minimum(all_p_fitness), mean(all_p_fitness), maximum(all_p_fitness)]
            debug.position_data[:,:,count] = PSO_.global_best.position
            debug.global_best_fitness[count] = PSO_.global_best.H

            debug.testing_vec[count] = test
            debug.confusion[count,:] = confusion
            debug.seeds[count] = seed_start + p
            count+=1
        end
    end

    dir = "$(debug.output_dir)/iteration$(iteration)"
    println(dir)
    if !isdir(dir)
        mkdir(dir)
    end

    io_seeds = open("$(dir)/seeds.txt", "w")
    io_fitness = open("$(dir)/fitness.txt", "w")
    writedlm(io_fitness, ["min_fitness" "mean_fitness" "max_fitness" "test_fitness" "KD_fitness" "global_best_fitness" "KD" "TP" "TN" "FP" "FN"]," ")
    
    io_parameters = open("$(dir)/params.txt", "w")
    write(io_parameters, "Vmax: $(Vₘₐₓ)\n")
    write(io_parameters, "W: $(W)\n")
    write(io_parameters, "phi1: $(φ₁)\n")
    write(io_parameters, "phi2: $(φ₂)\n")

    io_particle = Vector{IOStream}(undef, clusters[1].K)
    for k = 1:clusters[1].K
        io_particle[k] = open("$(dir)/cluster$(k).txt", "w")
    end
    # these can probably be written out without looping
    for i = 1:length(clusters)*num_seeds
        writedlm(io_seeds, debug.seeds[i])
        writedlm(io_fitness, hcat(debug.fitness_vec[i,:]', debug.testing_vec[i,:], KD_fitness[i,:], debug.global_best_fitness[i,:], KD[i,:], debug.confusion[i,:]', iterations[i,:]))
        for k = 1:clusters[1].K
            writedlm(io_particle[k], debug.position_data[:,k,i]', " ")
        end
    end

    for k = 1:clusters[1].K
        close(io_particle[k])
    end
    close(io_fitness)
    close(io_seeds)
    close(io_parameters)

    return mean(filter(!isnan,KD_fitness))
end

function WBCD_algorithm(clusters::cluster_data, DATA::TrainData, number_of_particles::Int, Vₘₐₓ::Float64, W::Float64, φ₁::Float64, φ₂::Float64; alpha::Float64=0.2, beta::Float64=0.4, gamma::Float64=0.4, debug::output, seed::Int=1337)
    #Random.seed!(seed)
    PSO_ = PSO(clusters, DATA, W=W, φ₁=φ₁, φ₂=φ₂, alpha=alpha, beta=beta, gamma=gamma, placement="random", M=number_of_particles, Vmax=Vₘₐₓ)

    println("Running particle swarm...")

    if !isdir("$(debug.output_dir)")
        mkdir("$(debug.output_dir)")
    end

    if debug.debug
        io = Vector{IOStream}(undef, PSO_.M)
        for m = 1:PSO_.M
            io[m] = open("$(debug.output_dir)/output$(m).txt", "w")
            write(io[m], "particleSize: $(size(PSO_.particles[m].position,1))\n")
            write(io[m], "W:$(PSO_.W), phi_1:$(PSO_.φ₁), phi_2:$(PSO_.φ₂)\n")
        end
    end

    fitness = Vector{Float64}(undef, PSO_.M)
    if debug.debug
        fitness_file = open("$(debug.output_dir)/fitness_$(seed).txt", "w")
    end

    mean_fitness = 0
    previous_mean_fitness = 0
    delta_fitness = 9999
    ε = 0.1
    p = 1

    buff_size = 20
    delta_buff = Vector{Float64}(undef, 20)
    count = 1
    #while (delta_fitness > ε || mean_fitness > 0.6 || p < 100) && p < 2000
    while (p < 200 || any(delta_buff .> ε)) && p < 2000
        #        println(delta_fitness," ",mean_fitness," ",p)
        update_PSO!(PSO_, data=DATA, ThresholdMethod="normal")
        for m = 1:PSO_.M
            fitness[m] = PSO_.particles[m].H
            if debug.debug
                writedlm(io[m], [p; PSO_.particles[m].H; vec(PSO_.particles[m].position); vec(PSO_.particles[m].velocity)]', ", ")
            end
        end
        if debug.debug
            writedlm(fitness_file, [minimum(fitness); maximum(fitness);
                mean(fitness); median(fitness);
                PSO_.global_best.H]')
        end
        mean_fitness = mean(map(x -> x.H, PSO_.particles))
        min_fitness = minimum(map(x -> x.H, PSO_.particles))
        max_fitness = maximum(map(x -> x.H, PSO_.particles))

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
    println(delta_fitness, " ", mean_fitness, " ", p)
    println("buffer: ", delta_buff)

    final_fitness = PSO_.global_best.H
    testing_fitness, confusion = PSO_fitness(PSO_.global_best, DATA, test_train="test", alpha=PSO_.alpha, beta=PSO_.beta, gamma=PSO_.gamma)
    println("Testing Set Accuracy: ", testing_fitness)

    if any(delta_buff .> ε)
        tmp = delta_buff .- ε
        tmp[tmp.<0] .= 0
        KD = 1 - maximum(abs.(tmp)) 
        final_fitness *= KD
    else
        KD = 1
    end
    #max_delta = maximum(delta_buff)
    #KD = max_delta * 4/10
    #KD = min(KD, 0.2)
    #if KD < 0
    #    KD = 0
    #end
    #final_fitness -= KD

    println("End training: ", final_fitness)
    # close all open files
    if debug.debug
        close(fitness_file)
        for m = 1:PSO_.M
            close(io[m])
        end
    end

    println("Done in ", p, " iterations")
    return final_fitness, KD, testing_fitness, confusion, PSO_, p
end

function majority_vote(cluster::cluster_data, DATA::WBCD_data, Vmax::Float64, W::Float64, phi1::Float64, phi2::Float64, alpha::Float64, beta::Float64, gamma::Float64)
    iterations = 100
    swarms = Array{PSO}(undef, iterations)
    # start by getting 100 particle swarms, initialized with 100 seeds
    p = 1
    debug = output(iterations, cluster.K, size(clusters[1].c,1)+size(clusters[1].std,1)+size(clusters[1].w,2), false)
    for i = 1:iterations
        if i == iterations
            debug.debug = true
        end
        TDAT = TrainData(DATA)
        _, _, _, _, swarms[i], _ = WBCD_algorithm(cluster, TDAT, 20, Vmax, W, phi1, phi2, alpha=alpha, beta=beta, gamma=gamma, debug=debug, seed=1337 + p)
        p += 1
    end

    TDAT = TrainData(DATA)
    # now evaluate the ANFIS network for each seed
    classifications = Array{Int,2}(undef, size(TDAT.testing_d,1), iterations)
    fitness = Array{Float64}(undef, iterations)
    PSOfitness = Array{Float64}(undef, iterations)
    for  i = 1:iterations
        c = swarms[i].global_best.position[1:size(cluster.c,1), :]
        σ = swarms[i].global_best.position[size(cluster.c,1) + 1:2*size(cluster.c,1), :]
        pq = swarms[i].global_best.position[2*size(cluster.c,1)+1:end-1,:]
        w = swarms[i].global_best.position[end, :]
        classifications[:, i] = Int.(calculate_NFS(c, σ, pq, w, TDAT.testing_x) .>= 0.5)
        TP, TN, FP, FN = calculate_fitness(classifications[:,i], TDAT.testing_d, α=alpha, β=beta, γ=gamma, ACC=true)
        fitness[i] = (TP + TN) / (TP + TN + FP + FN)
        PSOfitness[i] = swarms[i].global_best.H
    end

    vote = mode.(eachrow(classifications))
    TP, TN, FP, FN = calculate_fitness(vote, TDAT.testing_d, α=alpha, β=beta, γ=gamma, ACC=true)
    println(TP, " | ", FP)
    println(FN, " | ", TN)
    println(fitness)
    print("Maximum fitness: ")
    println(maximum(fitness))
    print("Best PSO -- testing: ")
    println(fitness[argmax(PSOfitness)])
    print("Vote accuracy: ")
    println((TP + TN)/(TP + TN + FP + FN))
    return [TP, TN, FP, FN]
end