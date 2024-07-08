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