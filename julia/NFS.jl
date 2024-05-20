export calculate_NFS

function calculate_NFS(clusters::cluster_data, x::Matrix{Float64}, d::Vector{Int})
    ŷ = Vector{Int}(undef, size(x,1))
    for i = 1:size(x,1)
        O² = exp.(-((x[i,:] .- clusters.c).^2)./(clusters.std).^2)
        O³ = prod.(eachcol(O²))'
        ŷ[i] = sum(O³.*clusters.w) < 0.5 ? 0 : 1
    end
    
    return ŷ
end

function calculate_fitness(guess::Vector{Int}, truth::Vector{Int}, α::Float64, β::Float64, γ::Float64)
    Accuracy = sum((guess .== truth)) / size(guess,1)
    TP = sum((guess .== 1.0) .& (truth .== 1.0))
    TN = sum((guess .== 0.0) .& (truth .== 0.0))
    FP = sum((guess .== 1.0) .& (truth .== 0.0))
    FN = sum((guess .== 0.0) .& (truth .== 1.0))

    Sensitivity = TP / (TP + FN)
    Specificity = TN / (FP + TN)

    return α * Accuracy + β * Sensitivity + γ * Specificity
end