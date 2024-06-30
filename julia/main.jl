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
function loadData(number_of_clusters::Int; filepath::String="./data/diagnostic.data")
    DATA = WBCD_data(filepath,nvars=10)
    #clusters = ClusterAnalysis.kmeans(DATA.training_x', number_of_clusters)
    clusters = cluster_data(DATA.training_x[:,1:10], DATA.training_d, number_of_clusters)
    println("Clusters generated...")
    return clusters, DATA
end

function WBCD_algorithm(clusters::cluster_data, DATA::WBCD_data, number_of_particles::Int, Vₘₐₓ::Float64, W::Float64, φ₁::Float64, φ₂::Float64; debug::Bool=false)
    Random.seed!(1337)
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
    fitness_file = open("outputs/fitness.txt", "w")
    previous_mean_fitness = mean(map(x->x.H, PSO_.particles))
    mean_fitness = 0
    delta_fitness = 9999
    ε = 1e-5
    p = 1
    while (delta_fitness > ε || mean_fitness > 0.6 || p < 100) && p < 2000
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
        delta_fitness = abs(mean_fitness - previous_mean_fitness)
        #println(p," ", delta_fitness)

        previous_mean_fitness = mean_fitness
        p = p + 1
    end

    testing_set_fitness = PSO_fitness(PSO_.global_best, DATA, test_train="test")
    println("Testing Set Fitness (not Acc): ", testing_set_fitness)

    # close all open files
    close(fitness_file)
    if debug
    for m = 1:PSO_.M
    close(io[m])
    end
    end

    println("Done in ", p, " iterations")
    return testing_set_fitness
end