struct particle
    velocity::Vector{Float64}
    position::Vector{Float64}
    position_best::Vector{Float64}

    function particle(velocity::Vector{Float64}, position::Vector{Float64}, position_best::Vector{Float64})
        new(velocity, position, position_best)
    end
end

struct PSO
    W::Float64 # Inertial weight
    φ₁::Float64 # Cognitive influence
    φ₂::Float64 # Social influence
    M::Int # Number of particles
    Vmax::Float64 # Maximum velocity
    global_best::Vector{Float64} # global best position
    particles::Vector{particle} # vector of particles

    function(cluster::cluster_data; W::Float64=1.0, φ₁::Float64=1.0, φ₂::Float64=1.0, M::Int=20, Vmax::Float64)

        particles = Vector{particle}(undef, M)
        for m = 1:M
            particles[m] = particle(velocity, position, position)
        end
        new(W, φ₁, φ₂i, M, global_best, particles)
    end
end

function PSO(clusters::cluster_data, data::WBCD_data)

end

function init_PSO(clusters::cluster_data)

end

function update_PSO()
end