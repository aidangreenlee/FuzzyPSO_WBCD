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
end

mutable struct PSO
    W::Float64 # Inertial weight
    φ₁::Float64 # Cognitive influence
    φ₂::Float64 # Social influence
    M::Int # Number of particles
    Vmax::Float64 # Maximum velocity
    alpha::Float64
    beta::Float64
    gamma::Float64
    global_best::particle
    particles::Vector{particle} # vector of particles


    function PSO(cluster::cluster_data, x::TrainData; W::Float64=1.0, φ₁::Float64=1.0, φ₂::Float64=1.0, M::Int=20, Vmax::Float64=3.0, alpha::Float64=0.2, beta::Float64=0.4, gamma::Float64=0.4, placement::String="clustered")
        particles = Vector{particle}(undef, M)
        H = Vector{Float64}(undef, M)
        P = Matrix{Float64}(undef,2*size(cluster.c,1),length(cluster.w))
        for m = 1:M

            # Set Positions
            if m == 1 || placement == "clustered" # always place particle 1 at clustered location
                P = vcat(cluster.c, cluster.std, cluster.w') # create particle position vector with positions and stdevs and output weight
            else
                P = vcat( rand(size(cluster.c,1),size(cluster.c,2)),
                          rand(size(cluster.std,1),size(cluster.std,2)),
                          rand(size(cluster.w,2),size(cluster.w,1))
                        )*20.0.-5 # set particle position randomly between 0 to 10 in each dimension
            end

            # Now set velocities
            if placement == "clustered"
                velocity = 2 * 0.1 * rand(size(P,1), size(P,2)) .- 0.1 # velocity is random between ±Vmax
            else
                velocity = zeros(size(P,1), size(P,2)) # initial velocity is 0
            end

            y = calculate_NFS(P[1:size(cluster.c,1),:], P[size(cluster.c,1)+1:end-1,:], cluster.w, x.training_x)
            H[m] = calculate_fitness(y, x.training_d)
            particles[m] = particle(velocity, P, H[m]) # initialize best position to the initial position
        end

        # get the particle with maximum fitness
        global_best = deepcopy(particles[argmax(H)])

        new(W, φ₁, φ₂, M, Vmax, alpha, beta, gamma, global_best, particles)
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
        new(W, φ₁, φ₂, M, Vmax, 0.2, 0.4, 0.4, global_best, particles)
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
            pso.particles[m].H = PSO_fitness(pso.particles[m], data, alpha=pso.alpha, beta=pso.beta, gamma=pso.gamma)
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

function PSO_fitness(p::particle, data::TrainData; test_train::String = "train", alpha::Float64=0.2, beta::Float64=0.4, gamma::Float64=0.4)
    n = Int((size(p.position, 1) - 1)/2)
    c = p.position[1:n, :]
    std = p.position[n+1:2*n, :]
    w = p.position[end, :]

    if test_train == "train"
        y = calculate_NFS(c, std, w, data.training_x)
        H = calculate_fitness(y, data.training_d)
    else
        y = calculate_NFS(c, std, w, data.testing_x)
        y = Int.(y .>= 0.5)
        TP, TN, FP, FN = calculate_fitness(y, data.testing_d, α=alpha, β=beta, γ=gamma, ACC=true)
        H = (TP + TN) / (TP + TN + FP + FN)
        println("Accuracy: ", H)
        println(TP," | ", FP)
        println(FN," | ", TN)
        return H, [TP, TN, FP, FN]
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