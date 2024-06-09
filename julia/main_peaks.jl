import ClusterAnalysis

include("./data.jl")
include("./cluster_data.jl")
include("./NFS.jl")
include("./PSO.jl")

PSO_ = PSO(W=.9,φ₁=1.0,φ₂=2.0)
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
for p = 1:300
    update_PSO!(PSO_)
    for m = 1:PSO_.M
    fitness[m] = PSO_.particles[m].H
    writedlm(io[m], [p; PSO_.particles[m].H;vec(PSO_.particles[m].position); vec(PSO_.particles[m].velocity)]',", ")
    end
    writedlm(fitness_file, [minimum(fitness); maximum(fitness);
                            mean(fitness);    median(fitness);
                            PSO_.global_best.H]');
end

# close all open files
close(fitness_file)
for m = 1:PSO_.M
close(io[m])
end

println("Done.")