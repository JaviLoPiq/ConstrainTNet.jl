
""" 
    Struct with training parameters
"""
struct TrainParams 
    learning_rate::Float32 
    num_training_steps::Int 
    noise_strength::Float32 # strength of noise in SGD
end

function TrainParams(learning_rate::Float32, num_training_steps::Int; noise_strength=0.0)
    return TrainParams(learning_rate, num_training_steps, noise_strength)
end

struct Dataset
    num_samples::Int 
    num_features::Int 
    data::Matrix{Int} # TODO : consider over Bool instead?
    probs::Vector{Float32}
    Dataset(data::Matrix{Int}, probs::AbstractVector{Float32}) = new(Base.size(data,1), Base.size(data,2), data, Float32.(probs))
end

function Dataset(data::Matrix{Int})
    return Dataset(data, [Float32(1/Base.size(data,1)) for _ in 1:Base.size(data,1)])
end

mutable struct EnvTN
    num_samples::Int 
    num_tensors::Int 
    cano_center::Int 
    env_tensors::Array{ITensor}
    log_norm_env_tensors::Array{Float64}
    EnvTN(num_tensors::Int, Dset::Dataset) = new(Dset.num_samples, num_tensors, 1, Array{ITensor}(undef, Dset.num_samples, num_tensors), zeros(Dset.num_samples, num_tensors))
end

function EnvTN(Dset::Dataset) 
    return EnvTN(Dset.num_features, Dset)
end

# TODO: add regularization term in 1-site DMRG? remove kwargs... 

"""
    Training algorithm via 1-site DMRG.
"""
function training(train_params::TrainParams,
    mps::MPS,
    data_samples::Matrix{Int};
    kwargs...)

    normalize!(mps)
    constrained_orthogonalize!(mps, 1; kwargs...)
    sites_mps = siteinds(mps) 
    training_data = Dataset(data_samples)
    env_tensors = env_tensors_initial(mps, training_data)

    nll_vector = []
    for t in 1:train_params.num_training_steps
        mps = sweeping_update!(env_tensors, mps, sites_mps, training_data, train_params; kwargs...)
        append!(nll_vector, NLL_loss(env_tensors, mps, sites_mps, training_data))
    end 
    return nll_vector, mps
end

function env_tensors_initial(mps::MPS, dset::Dataset)
    @assert orthocenter(mps) == 1
    num_env_tensors = length(mps)
    env_tensors = EnvTN(dset)
    env_tensors.cano_center = 1

    sites_mps = siteinds(mps)
    Threads.@threads for sample_index in 1:dset.num_samples 
        env_tensors.env_tensors[sample_index, env_tensors.cano_center] = ITensor()
    end
    Threads.@threads for sample_index in 1:dset.num_samples 
        env_tensors.env_tensors[sample_index, num_env_tensors] = mps[num_env_tensors] * 
            dag(state(sites_mps[num_env_tensors], dset.data[sample_index, num_env_tensors] + 1))
        for v in num_env_tensors-1:-1:2    
            env_tensors.env_tensors[sample_index, v] = mps[v] * dag(state(sites_mps[v], dset.data[sample_index, v] + 1)) * 
                env_tensors.env_tensors[sample_index, v + 1]
        end
    end
    return env_tensors
end

function sweeping_update!(env_tensors::EnvTN, mps::MPS, sites_mps::Vector{Index{Vector{Pair{QN, Int64}}}}, dset::Dataset, train_params::TrainParams; kwargs...)
    for v in Iterators.flatten([1:length(mps), reverse(1:length(mps)-1)])
        constrained_orthogonalize!(mps, v; kwargs...)
        env_tensors_update!(env_tensors, mps, sites_mps, v, dset) 
        grad_mps = one_site_gradient(env_tensors, mps, sites_mps, v, dset)
        mps[v] -= train_params.learning_rate * grad_mps 
        normalize!(mps)
    end
    return mps 
end

function env_tensors_update!(env_tensors::EnvTN, mps::MPS, sites_mps::Vector{Index{Vector{Pair{QN, Int64}}}}, final_site::Int, dset::Dataset)
    @assert orthocenter(mps) == final_site
    if env_tensors.cano_center < final_site 
        for v in env_tensors.cano_center:final_site-1 
            Threads.@threads for sample_index in 1:dset.num_samples 
                env_tensors.env_tensors[sample_index, v] = mps[v]
                if v > 1 
                    env_tensors.env_tensors[sample_index, v] *= env_tensors.env_tensors[sample_index, v-1]
                end
                env_tensors.env_tensors[sample_index, v] *= dag(state(sites_mps[v], dset.data[sample_index, v] + 1))
            end
        end
    end 
    N = length(mps)
    if env_tensors.cano_center > final_site 
        for v in reverse(env_tensors.cano_center:final_site+1)
            Threads.@threads for sample_index in 1:dset.num_samples 
                env_tensors.env_tensors[sample_index, v] = mps[v]
                if v < N
                    env_tensors.env_tensors[sample_index, v] *= env_tensors.env_tensors[sample_index, v+1]
                end 
                env_tensors.env_tensors[sample_index, v] *= dag(state(sites_mps[v], dset.data[sample_index, v] + 1))
            end
        end
    end
    env_tensors.cano_center = final_site
end

function one_site_gradient(env_tensors::EnvTN, mps::MPS, sites_mps::Vector{Index{Vector{Pair{QN, Int64}}}}, v::Int, dset::Dataset)
    second_term_arr = [ITensor() for _ in 1:Threads.nthreads()]
    N = length(mps)
    Threads.@threads for sample_index in 1:dset.num_samples
        if v == 1
            mps_prime = env_tensors.env_tensors[sample_index, 2] * dag(state(sites_mps[1], dset.data[sample_index,1] + 1))
        elseif v == N
            mps_prime = env_tensors.env_tensors[sample_index, N-1] * dag(state(sites_mps[N], dset.data[sample_index,N] + 1))
        else 
            mps_prime = env_tensors.env_tensors[sample_index, v-1] * dag(state(sites_mps[v], dset.data[sample_index,v] + 1)) * 
                env_tensors.env_tensors[sample_index, v+1]
        end
        mps_ = (mps[v] * mps_prime)[]
        if mps_ != 0 
            pterm = 1/dset.num_samples * mps_prime / mps_ 
            second_term_arr[Threads.threadid()] += pterm
        end
    end
    second_term = sum(second_term_arr)
    return mps[v] - 2 * dag(second_term) 
end

function NLL_loss(env_tensors::EnvTN, mps::MPS, sites_mps::Vector{Index{Vector{Pair{QN, Int64}}}}, dset::Dataset)
    cano_center = orthocenter(mps)
    env_tensors_update!(env_tensors, mps, sites_mps, cano_center, dset)
    nll = 0 
    for sample_index in 1:dset.num_samples 
        if cano_center == 1
            ampl = mps[cano_center] * dag(state(sites_mps[cano_center], dset.data[sample_index, cano_center] + 1)) * env_tensors.env_tensors[sample_index, 2]
        elseif cano_center == N 
            ampl = mps[cano_center] * dag(state(sites_mps[cano_center], dset.data[sample_index, cano_center] + 1)) * env_tensors.env_tensors[sample_index, N-1]
        else 
            ampl = mps[cano_center] * dag(state(sites_mps[cano_center], dset.data[sample_index, cano_center] + 1)) *
                env_tensors.env_tensors[sample_index, cano_center-1] * env_tensors.env_tensors[sample_index, cano_center+1]
        end
        nll -= 1/dset.num_samples * log(abs(ampl[])^2)
    end     
    return nll 
end

function initialization(A::Matrix{Int}, lb::Vector{Int}, ub::Vector{Int}; ortho_center=1, block_dim=1)
    N = length(A[1,:])
    mps, link_inds_backward, link_inds_forward = constraints_to_mps(A, lb, ub; verbose=true, ortho_center, block_dim)
    return mps, link_inds_backward, link_inds_forward
end


function ind2state(x::Int, len::Int)
    bs_ = bitstring(x)
    bs = collect(bs_[end-len+1:end])
    bs .== '1'
end

function linear_cost_function(w::Vector{T}, v::Vector{Int}) where T <: Union{Float64, Int64}
    return dot(w, v)
end

function quadratic_cost_function(Q::Matrix{T}, v::Vector{Int}) where T <: Union{Float64, Int64}
    return transpose(v) * Q * v 
end

function sample_from_dict(dict::Dict{Vector{Int}, Float64})
    bitstrings = collect(keys(dict))
    probabilities = collect(values(dict))

    rand_num = rand()

    # Find the index of the first cumulative probability greater than the random number
    index = searchsortedfirst(cumsum(probabilities), rand_num)

    return bitstrings[index]
end


function extract_training_samples(dict::Dict{Vector{Int}, Float64}, num_samples::Int, num_features::Int)
    training_samples = Matrix{Int}(undef, num_samples, num_features)
    for i in 1:num_samples 
        training_samples[i,:] = sample_from_dict(dict)
    end
    return training_samples 
end

function boltzmann_probabilities(cost_function, bitstrings::Vector{Vector{Int}}, temperature::Float64)

    if temperature == Inf 
        weights = [1. for _ in 1:length(bitstrings)]
    else 
        energies = [cost_function(bitstring) for bitstring in bitstrings]
        weights = exp.(-energies / temperature)
    end
    weights /= sum(weights)

    probability_dict = Dict(bitstrings[i] => weights[i] for i in 1:length(bitstrings))

    return probability_dict
end

function optimizer(cost_function, A::Matrix{Int}, lb::Vector{Int}, ub::Vector{Int}, mps_ini::MPS, link_inds_forward::Vector{Index}, 
    link_inds_backward::Vector{Index}, flux_mps::QRegion, num_loops::Int; verbose::Bool=false, kwargs...) 
    mps = copy(mps_ini)
    num_train_samples = 400
    N = length(mps)
    learning_rate::Float32 = get(kwargs, :learning_rate, Float32(0.5E-1))
    num_training_steps::Int = get(kwargs, :num_training_steps, 1)
    dict_samples = Dict{Vector{Int}, Float64}()
    history_costs = Vector{Vector{Union{Int,Float64}}}()
    total_data_samples = Set{Vector{Int}}()
    seed_samples = Vector{Vector{Int}}()
    new_costs = Vector{Union{Int,Float64}}()
    for _ in 1:num_train_samples 
        seed_sample = sample(mps) .- 1 
        push!(seed_samples, seed_sample)
        push!(total_data_samples, seed_sample)
        push!(new_costs, cost_function(seed_sample))
    end
    push!(history_costs, new_costs)
    cum_minimum = minimum(new_costs)
    min_old_costs = 0
    for i in 1:num_loops 
        println("Loop ", i, " minimum so far ", cum_minimum)
        sorted_costs = sort([cost_function(v) for v in seed_samples])
        #temperature = 500/i # N = 200
        #temperature = 1000/i # N = 400 
        #temperature = 250/i # N = 100
        temperature = 125/i # N = 50
        dict_samples = boltzmann_probabilities(cost_function, collect(total_data_samples), temperature)
        training_samples = extract_training_samples(dict_samples, num_train_samples, N)
        train_params = TrainParams(learning_rate, num_training_steps)
        if i > 1 
            if minimum(new_costs) >= min_old_costs #cum_minimum  
                rows_to_delete = randperm(size(training_samples, 1))[1:Int(0.1*num_train_samples)]
                training_samples = training_samples[setdiff(1:end, rows_to_delete), :]
                mps = copy(mps_ini)
                for _ in 1:Int(0.1*num_train_samples)
                    training_samples = vcat(training_samples, transpose(sample(mps_ini) .- 1))
                end
            else 
                if minimum(new_costs) < cum_minimum 
                    cum_minimum = minimum(new_costs)
                end
            end
        end
        _, mps = training(train_params, mps, training_samples; left_canonical_indices=link_inds_forward, right_canonical_indices=link_inds_backward, flux_mps=flux_mps, verbose=verbose, kwargs...)

        min_old_costs = minimum(new_costs) 
        new_costs = Vector{Union{Int, Float64}}()
        seed_samples = Vector{Vector{Int}}()
        for i in 1:num_train_samples 
            seed_sample = sample(mps) .- 1 
            push!(seed_samples, seed_sample)
            push!(total_data_samples, seed_sample)
            push!(new_costs, cost_function(seed_sample))
        end
        push!(history_costs, new_costs)
    end
    _, opt_bitstring = findmax(dict_samples)
    opt_cost = cost_function(opt_bitstring) 
    @assert all(lb .<= A * opt_bitstring .<= ub)
    return opt_cost, opt_bitstring, mps, history_costs
end