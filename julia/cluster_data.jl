using Statistics
export cluster

mutable struct cluster_data
    c::Matrix{Float64}
    std::Matrix{Float64}
    w::Vector{Float64}
    K::Int
    N::Vector{Int}
    
    function cluster_data(data::Matrix{Float64}, diagnosis::Vector{Int}, number_of_clusters::Int)
        clusters = ClusterAnalysis.kmeans(data, number_of_clusters)
        c = reduce(hcat, clusters.centroids)
        std = reduce(hcat, map(r->Statistics.std.(eachcol(data[clusters.cluster.==r,:])), 1:number_of_clusters))
        w = map(r->Statistics.mean((diagnosis[clusters.cluster.==r])), 1:number_of_clusters)
        N = map(r->sum(clusters.cluster.==r),1:5)
        new(c, std, w, number_of_clusters, N)
    end
end