using Random
function run_job(seed)
    N = 100

    Random.seed!(seed)
    A = rand(0:5, 1, N)
    lb = [0]
    ub = [Int(floor(N/4))]

    flux_mps = QRegion([Box(lb, ub)])
    mps, link_inds_backward, link_inds_forward = constraints_to_mps(A, lb, ub; verbose=false, flux_center=1, block_dim=1)

    constrained_orthogonalize!(mps, 1; left_canonical_indices=link_inds_forward, right_canonical_indices=link_inds_backward, flux_mps=flux_mps, verbose=false)
    normalize!(mps)
    mps_ini = copy(mps)

    Random.seed!(seed)
    Q = rand(-5:5, N, N)
    cost_function = v -> quadratic_cost_function(Q, v)
    opt_cost, opt_bitstring, mps, history_costs = optimizer(cost_function, A, lb, ub, mps_ini, link_inds_forward, link_inds_backward, flux_mps, 75; cutoff=Float32(1e-4), min_blockdim=0, verbose=false)
    return history_costs
end

seeds = [1, 2, 3, 4, 5]
results = Vector{Vector{Vector{Int64}}}(undef, length(seeds))

for i in eachindex(seeds)
    println("seed: ", i)
    @time results[i] = run_job(seeds[i])
end

#using Serialization
#serialize("results_N_100.jls", results)