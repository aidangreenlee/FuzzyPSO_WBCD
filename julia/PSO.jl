using LinearAlgebra

mutable struct particle
    velocity::Matrix{Float64}
    position::Matrix{Float64}
    position_best::Matrix{Float64}
    H::Float64 # fitness
    Hbest::Float64 # best fitness

    function particle(velocity::Matrix{Float64}, position::Matrix{Float64}, H::Float64)
        new(velocity, position, position, H, H)
    end

    function particle(velocity::Array{Float64,3}, position::Array{Float64,3}, H::Array{Float64})
        P = Array{particle}(undef, size(velocity,3))
        for i = 1:size(velocity,3)
            P[i] = particle(velocity[:,:,i], position[:,:,i], H[i])
        end
        return P
    end
end

mutable struct PSO
    W::Float64 # Inertial weight
    φ₁::Float64 # Cognitive influence
    φ₂::Float64 # Social influence
    M::Int # Number of particles
    Vmax::Float64 # Maximum velocity
    global_best::particle
    particles::Vector{particle} # vector of particles


    # Inputs:
    # cluster: Array of N clusters associated with N training sets
    # x:       Array of N training sets
    function PSO(cluster::Array{cluster_data}, x::Array{WBCD_data}; W::Float64=1.0, φ₁::Float64=1.0, φ₂::Float64=1.0, M::Int=20, Vmax::Float64=3.0, placement::String="clustered")
        #particles = Vector{particle}(undef, M)
        particles = Array{particle}(undef, M, length(cluster))
        #H = Vector{Float64}(undef, M)
        H = Array{Float64}(undef, M, length(cluster))
        #P = Matrix{Float64}(undef,2*size(cluster.c,1),length(cluster.w))
        P = Array{Float64}(undef, 2*size(cluster[1].c,1), length(cluster[1].w), length(cluster))
        for m = 1:M

            # Set Positions
            if m == 1 || placement == "clustered" # always place particle 1 at clustered location
                # C = map(x->x.c, cluster )
                C = reshape(reduce(hcat,map(x->x.c,cluster)), length(cluster), cluster[1].K, length(cluster))
                # S = map(x->x.std, cluster)
                S = reshape(reduce(hcat,map(x->x.std,cluster)), length(cluster), cluster[1].K, length(cluster))
                # W = map(x->reshape(x.w,1,cluster[1].K), cluster)
                W = reshape(reduce(vcat,map(x->reshape(x.w,1,cluster[1].K), cluster)),1, cluster[1].K, length(cluster))
                #P = map(x->vcat(x[1],x[2],x[3]),A)
                P = vcat(C,S,W)
                # P = vcat(cluster.c, cluster.std, cluster.w') # create particle position vector with positions and stdevs and output weight
            else
                P = rand(size(cluster[1].c,1) + size(cluster[1].std,1) + size(cluster[1].w,2), cluster[1].K, length(cluster))
            end

            # Now set velocities
            if placement == "clustered"
                velocity = 2 * 0.1 * rand(size(P,1), size(P,2), size(P,3)) .- 0.1 # velocity is random between ±Vmax
            else
                velocity = zeros(size(P,1), size(P,2), size(P,3)) # initial velocity is 0
            end
            
            temp = map(x->x.training_x, x) 
            NFS_input = reduce((x,y)->cat(x,y,dims=3),(temp[i] for i in 1:length(temp)))
            #y = calculate_NFS(P[1:size(cluster.c,1),:], P[size(cluster.c,1)+1:end-1,:], cluster.w, x.training_x)
            # y = calculate_NFS(P[1:size(cluster.c,1),:], P[size(cluster.c,1)+1:end-1,:], cluster.w, x.training_x)
            y = calculate_NFS(P, NFS_input)
            truth = reduce(hcat,map(x->x.training_d, x))
            H[m,:] = calculate_fitness(y, truth)
            particles[m,:] = particle(velocity, P, H[m,:]) # initialize best position to the initial position
        end

        # get the particle with maximum fitness
        global_best = deepcopy(particles[argmax(H)])

        new(W, φ₁, φ₂, M, Vmax, global_best, particles)
    end

    function PSO(; W::Float64=1.0, φ₁::Float64=1.0, φ₂::Float64=1.0, M::Int=20, Vmax::Float64=0.2)
        particles = Vector{particle}(undef, M)
        for m = 1:M
            P = Matrix([0. 0.]')
            #P = rand(2,1) * 3.0 .- 3.0
            velocity = (rand(2,1) * 2 .- 1)*0.1
            H = peaks(P[1],P[2]) + 8
            particles[m] = particle(velocity, P, H)
        end
        global_best = deepcopy(particles[1])
        new(W, φ₁, φ₂, M, Vmax, global_best, particles)
    end
end

function update_PSO!(pso::PSO; data::Any = nothing, ThresholdMethod::String = "element-wise")
    for m = 1:pso.M
        # 1. Update velocity
        R¹ = rand(size(pso.particles[m].position, 1), size(pso.particles[m].position, 2))
        R² = rand(size(pso.particles[m].position, 1), size(pso.particles[m].position, 2))
        #R² = R² * 0.0 .+ 1
        #R¹ = R¹ * 0.0 .+ 0.01
        pso.particles[m].velocity = pso.W * pso.particles[m].velocity +
            pso.φ₁ * R¹ .* (pso.particles[m].position_best .- pso.particles[m].position) + 
            pso.φ₂ * R² .* (pso.global_best.position .- pso.particles[m].position)

        # 2. Apply velocity threshold
        if ThresholdMethod == "normal"
            velMag = norm.(eachcol((pso.particles[m].velocity)))'
            for i_mag = eachindex(velMag)
                if velMag[i_mag] > pso.Vmax
                    v̂ = pso.particles[m].velocity[:,i_mag] ./ velMag[i_mag]
                    pso.particles[m].velocity[:,i_mag] = v̂ * pso.Vmax
                end
            end
        elseif ThresholdMethod == "element-wise"
            overMax = abs.(pso.particles[m].velocity) .> pso.Vmax
            pso.particles[m].velocity[overMax] .= pso.Vmax .* sign.(pso.particles[m].velocity[overMax])
        elseif ThresholdMethod == "none"

        else
            error("Not A Valid Vmax ThresholdMethod")
        end

        # 3. Update particle position
        pso.particles[m].position = pso.particles[m].position + pso.particles[m].velocity

        # 4. Update fitness
        if isnothing(data)
            pso.particles[m].H = PSO_fitness(pso.particles[m])
        else
            pso.particles[m].H = PSO_fitness(pso.particles[m], data)
        end

        # 5. Update global particle best
        if pso.particles[m].H > pso.global_best.H
            pso.global_best = deepcopy(pso.particles[m])
            #println(pso.particles[m].H)
        end

        # 6. Update particle personal best
        if pso.particles[m].H > pso.particles[m].Hbest
            #println("new best")
            pso.particles[m].position_best = copy(pso.particles[m].position)
            pso.particles[m].Hbest = pso.particles[m].H
        end
    end
end

function PSO_fitness(p::particle, data::WBCD_data; test_train::String = "train")
    n = Int((size(p.position, 1) - 1)/2)
    c = p.position[1:n, :]
    std = p.position[n+1:2*n, :]
    w = p.position[end, :]

    if test_train == "train"
        y = calculate_NFS(c, std, w, data.training_x)
        H = calculate_fitness(y, data.training_d)
    else
        y = calculate_NFS(c, std, w, data.testing_x)
        TP, TN, FP, FN = calculate_fitness(y, data.testing_d, ACC=true)
        H = (TP + TN) / (TP + TN + FP + FN)
        println("Accuracy: ", H)
        println(TP," | ", FP)
        println(FN," | ", TN)
        return H
    end

    return H
end

function PSO_fitness(p::particle)
    return 8 + peaks(p.position[1],p.position[2])
end

function peaks(x::Float64,y::Float64)
    z = 3*(1-x).^2 .* exp(-(x.^2) - (y+1).^2) -
        10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) -
        1/3*exp(-(x+1).^2 - y.^2)
   return z
end