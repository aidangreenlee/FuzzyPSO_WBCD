import ClusterAnalysis

include("./data.jl")
include("./cluster_data.jl")
include("./NFS.jl")
include("./PSO.jl")

DATA = WBCD_data("./data/diagnostic.data")

number_of_clusters = 5

clusters = ClusterAnalysis.kmeans(DATA.training_x', number_of_clusters)
clusters = cluster_data(DATA.training_x, DATA.training_d, 3)
y = calculate_NFS(clusters, DATA.training_x, DATA.training_d)

println(sum(y))
