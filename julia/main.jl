import ClusterAnalysis
import Random
Random.seed!(1337)

include("./data.jl")
include("./cluster_data.jl")
include("./NFS.jl")
include("./PSO.jl")

DATA = WBCD_data("./data/diagnostic.data",nvars=10)

number_of_clusters = 5

#clusters = ClusterAnalysis.kmeans(DATA.training_x', number_of_clusters)
clusters = cluster_data(DATA.training_x[:,1:10], DATA.training_d, number_of_clusters)
println("Clusters generated...")
PSO_ = PSO(clusters, DATA,W=1.00,φ₁=1.4,φ₂=1.5,placement="random",M=200, Vmax=3.)

println("Running particle swarm...")

if !isdir("outputs")
    mkdir("outputs")
end

io = Vector{IOStream}(undef, PSO_.M)
for m = 1:PSO_.M
io[m] = open("outputs/output$(m).txt","w")
write(io[m], "particleSize: $(size(PSO_.particles[m].position,1))\n")
write(io[m], "W:$(PSO_.W), phi_1:$(PSO_.φ₁), phi_2:$(PSO_.φ₂)\n")
end

fitness = Vector{Float64}(undef, PSO_.M)
fitness_file = open("outputs/fitness.txt", "w")
for p = 1:500
    update_PSO!(PSO_, data=DATA, ThresholdMethod="normal")
    for m = 1:PSO_.M
    fitness[m] = PSO_.particles[m].H
    writedlm(io[m], [p; PSO_.particles[m].H;vec(PSO_.particles[m].position); vec(PSO_.particles[m].velocity)]',", ")
    end
    writedlm(fitness_file, [minimum(fitness); maximum(fitness);
                            mean(fitness);    median(fitness);
                            PSO_.global_best.H]');
end

println("Testing Set Fitness (not Acc): ", PSO_fitness(PSO_.global_best, DATA, test_train="test"))

# close all open files
close(fitness_file)
for m = 1:PSO_.M
close(io[m])
end

println("Done.")