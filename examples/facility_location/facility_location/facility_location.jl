using Revise
using Random
using ProfileView
using Serialization 
include("constraint_handling.jl")
include("constrained_optimization.jl")

function generate_matrix_A(num_facilities, num_demand_points, coverage_ratio)
    A = zeros(Int, num_demand_points, num_facilities)
    facilities_per_demand = ceil(Int, coverage_ratio * num_facilities)

    for j in 1:num_demand_points
        # Randomly select a subset of facilities for each demand point
        facilities = randperm(num_facilities)[1:facilities_per_demand]
        for f in facilities
            A[j, f] = 1
        end
    end
    
    return A
end

function complexity_facility_location(N::Int, M::Int, lower_bound::Int, upper_bound::Int, coverage_ratio::Float64, seed::Int)
    Random.seed!(seed)
    A = generate_matrix_A(N, M, coverage_ratio)
    ub = ones(Int, M) * upper_bound
    lb = ones(Int, M) * lower_bound
    start_time = time_ns()
    mps, link_inds_backward, link_inds_forward, max_num_qregions = constraints_to_mps(A, lb, ub; verbose=true, flux_center=1, block_dim=1)
    end_time = time_ns()
    elapsed_time = (end_time - start_time) / 1e9
    dims_link_inds_backward = [length(link_inds_backward[i].space) for i in eachindex(link_inds_backward)]
    max_num_qregions_backward = maximum(dims_link_inds_backward)
    dims_link_inds_forward = [length(link_inds_forward[i].space) for i in eachindex(link_inds_forward)]
    max_num_qregions_forward = maximum(dims_link_inds_forward)
    effective_num_qregions = maximum([max_num_qregions_backward, max_num_qregions_forward])
    return effective_num_qregions, max_num_qregions, elapsed_time, dims_link_inds_backward, dims_link_inds_forward
end

for N in [30,40,50,60,70,80,90,100]
    lower_bound = 2 
    upper_bound = 10
    coverage_ratio = 0.1 
    seed = 5

    dict = Dict()

    for M in [1,3,6]
        @show M
        effective_num_qregions, max_num_qregions, elapsed_time, dims_link_inds_backward, dims_link_inds_forward = complexity_facility_location(N, M, lower_bound, upper_bound, coverage_ratio, seed)
        dict[M] = Dict("effective_num_qregions" => effective_num_qregions, "max_num_qregions" => max_num_qregions, "elapsed_time" => elapsed_time, "dims_link_inds_backward" => dims_link_inds_backward, "dims_link_inds_forward" => dims_link_inds_forward)
        serialize("facility_data_N=$(N)_lower_bound=$(lower_bound)_upper_bound=$(upper_bound)_coverage_ratio=$(coverage_ratio)_seed=$(seed).jls", dict)
    end
end
