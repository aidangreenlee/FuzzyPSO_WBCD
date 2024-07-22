export calculate_NFS

function calculate_NFS(c::Matrix{Float64}, std::Matrix{Float64}, w::Vector{Float64}, x::Matrix{Float64})
    ŷ = Vector{Int}(undef, size(x,1))
    for i = 1:size(x,1)
        O² = exp.(-((x[i,:] .- c).^2)./(std).^2)
        O³ = prod.(eachcol(O²))'
        ŷ[i] = sum(O³.*w) < 0.5 ? 0 : 1
    end
    
    return ŷ
end

function calculate_NFS(P::Array{Float64}, NFS_input::Array{Float64})
    # Inputs:
    # P: particle_position_dim x num_clusters x num_sets vector
    # NFS_input: num_datapoints x num_dimensions x num_sets
    y = Array{Int}(undef, size(NFS_input,1), size(NFS_input,3))
    sizeofP = Int((size(P,1) - 1)/2)
    c = P[1:sizeofP,:,:]
    s = P[sizeofP+1:2*sizeofP,:,:]
    w = P[end,:,:]

    # need to find a way to get rid of the loops here
    for j = 1:size(NFS_input,1)
        for i = 1:10
            O2 = exp.(-((NFS_input[j,:,i] .- c[:,:,i]).^2)./(s[:,:,i]).^2)
            O3 = prod.(eachcol(O2))'
            y[j,i] = sum(O3.*w[:,i]') < 0.5 ? 0 : 1
        end
    end
    return y
end

function calculate_fitness(guess::Vector{Int}, truth::Vector{Int}; α::Float64=0.2, β::Float64=0.4, γ::Float64=0.4, ACC::Bool=false)
    Accuracy = sum((guess .== truth)) / size(guess,1)
    TP = sum((guess .== 1.0) .& (truth .== 1.0))
    TN = sum((guess .== 0.0) .& (truth .== 0.0))
    FP = sum((guess .== 1.0) .& (truth .== 0.0))
    FN = sum((guess .== 0.0) .& (truth .== 1.0))

    Sensitivity = TP / (TP + FN)
    Specificity = TN / (FP + TN)

    if ACC
        return [TP, TN, FP, FN]
    else
        return α * Accuracy + β * Sensitivity + γ * Specificity
    end
end

function calculate_fitness(guess::Array{Int,2}, truth::Array{Int,2}; α::Float64=0.2, β::Float64=0.4, γ::Float64=0.4, ACC::Bool=false)
    Accuracy = sum.(eachcol(guess.==truth))./size(guess,1)

    TP = sum.(eachcol((guess .== 1.0) .& (truth .== 1.0)))
    TN = sum.(eachcol((guess .== 0.0) .& (truth .== 0.0)))
    FP = sum.(eachcol((guess .== 1.0) .& (truth .== 0.0)))
    FN = sum.(eachcol((guess .== 0.0) .& (truth .== 1.0)))

    Sensitivity = TP ./ (TP + FN)
    Specificity = TN ./ (FP + TN)

    if ACC
        return [TP, TN, FP, FN]
    else
        return α * Accuracy + β * Sensitivity + γ * Specificity
    end
end