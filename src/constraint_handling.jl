"""
Compute cumulative boundary region. 
"""
function boundaryregion(A::Matrix{Int})::Vector{QRegion} 
    m, N = size(A)
    lb_cumsum = Vector{Vector{Int}}(undef, N)
    ub_cumsum = Vector{Vector{Int}}(undef, N)
    qregion_vec = Vector{QRegion}()
    lb_cumsum[1] = zeros(Int,m)
    ub_cumsum[1] = zeros(Int,m)
    for j in 1:m
        if A[j,1] <= 0 
            lb_cumsum[1][j] = A[j,1]
            ub_cumsum[1][j] = 0 
        else 
            lb_cumsum[1][j] = 0 
            ub_cumsum[1][j] = A[j,1]
        end
    end
    push!(qregion_vec, QRegion([Box(lb_cumsum[1], ub_cumsum[1])]))
    for i in 2:N
        lb_cumsum[i] = zeros(Int,m)
        ub_cumsum[i] = zeros(Int,m)
        for j in 1:m
            if A[j,i] <= 0 
                lb_cumsum[i][j] = lb_cumsum[i-1][j] + A[j,i]
                ub_cumsum[i][j] = ub_cumsum[i-1][j]
            else 
                lb_cumsum[i][j] = lb_cumsum[i-1][j]
                ub_cumsum[i][j] = ub_cumsum[i-1][j] + A[j,i]
            end
        end
        push!(qregion_vec, QRegion([Box(lb_cumsum[i], ub_cumsum[i])]))
    end
    return qregion_vec
end

"""
Given some constraints 

lb <= A x <= ub, 

outputs :

- link_indices_forward : link indices as Vector{Vector{QRegion}} format.
- blocks_mps : set of nonzero blocks.
- max_num_qregions : maximum number of qregions found on backward sweep. 
"""

function constraints_to_indices(A::Matrix{Int}, lb::Vector{Int}, ub::Vector{Int}; verbose=false)
    N = length(A[1,:])

    boundaries_vector = boundaryregion(A) 

    link_indices_backward = Vector{Vector{QRegion}}() 
    push!(link_indices_backward, [QRegion([Box(lb,ub)])])
    link_indices_backward[1] = link_indices_backward[1] ∩ [boundaries_vector[N]]

    max_num_qregions = 0
    # backward sweep 
    for i in 2:N

        old_index = link_indices_backward[i-1] ∩ [boundaries_vector[N-i+1]]
        added_index = (link_indices_backward[i-1]-A[:,N-i+2]) ∩ [boundaries_vector[N-i+1]]

        if old_index === nothing && added_index === nothing 
            error("No feasiable solutions found.")
        end
        new_index = Vector{QRegion}()
        inter = old_index ∩ added_index 
        if inter !== nothing 
            for j in eachindex(inter)
                if inter[j] !== nothing 
                    append!(new_index, [inter[j]])
                end
            end
        end
        differ = old_index - added_index 
        for j in eachindex(differ)
            append!(new_index, [differ[j]])
        end
        
        push!(link_indices_backward, new_index)
        
        # TODO : prescind of this for getting rid of QRegion([]). Make sure this doesn't happen in difference
        link_indices_backward[i] = link_indices_backward[i] ∩ [boundaries_vector[N-i+1]]
        num_qregions = length(link_indices_backward[i])
        if verbose
            num_boxes = 0
            for j in eachindex(link_indices_backward[i])
                num_boxes += length(link_indices_backward[i][j].boxes)
            end 
            if num_qregions > max_num_qregions 
                max_num_qregions = num_qregions
            end
        end
    end 

    link_indices_forward = Vector{Vector{QRegion}}()
    blocks_mps = []
    blocks_site = Vector{Block{2}}() # TODO : we assume the structure of each Block of the form (i,j,k) with i site index, j left index, k right index (for 3-index tensors)

    new_index = Vector{QRegion}()
    qregion = interior([0 for i in 1:length(lb)], link_indices_backward[N])
    if qregion !== nothing 
        push!(new_index, qregion)
        push!(blocks_site, Block(1,1))
    end    
    qregion = interior(A[:,1], link_indices_backward[N])
    if qregion !== nothing 
        if qregion ∉ new_index 
            push!(new_index, qregion)
            push!(blocks_site, Block(2,2))
        else 
            push!(blocks_site, Block(2,1))
        end 
    end
    push!(link_indices_forward, new_index)
    append!(blocks_mps, [blocks_site])

    # forward sweep 
    for i in 2:N-1
        if verbose; @show i; end 
        blocks_site = Vector{Block{3}}()
        new_index = Vector{QRegion}()
        ind_left = 1
        for element in link_indices_forward[i-1] 
            qregion = interior(element, link_indices_backward[N-i+1])
            if qregion !== nothing
                if qregion ∉ new_index 
                    push!(new_index, qregion) 
                    ind_right = length(new_index)
                else
                    ind_right = findfirst(x->x==qregion, new_index)
                end
                push!(blocks_site, Block(1,ind_left,ind_right))
            end
            qregion = interior((element + A[:,i]), link_indices_backward[N-i+1])
            if qregion !== nothing
                if qregion ∉ new_index 
                    push!(new_index, qregion) 
                    ind_right = length(new_index)
                else 
                    ind_right = findfirst(x->x==qregion, new_index) 
                end
                push!(blocks_site, Block(2,ind_left,ind_right))
            end  
            ind_left += 1
        end
        push!(link_indices_forward, new_index)
        if verbose; @show i, length(link_indices_forward[i]); end
        append!(blocks_mps, [blocks_site])
    end

    ind_left = 1
    blocks_site = Vector{Block{2}}()
    new_index = Vector{QRegion}()
    for element in link_indices_forward[N-1]
        qregion = interior(element, link_indices_backward[1])
        if qregion !== nothing
            if qregion ∉ new_index 
                push!(new_index, qregion) 
                ind_right = length(new_index)
            else
                ind_right = findfirst(x->x==qregion, new_index)
            end
            push!(blocks_site, Block(1,ind_left))
        end
        qregion = interior((element + A[:,N]), link_indices_backward[1])
        if qregion !== nothing
            if qregion ∉ new_index 
                push!(new_index, qregion) 
                ind_right = length(new_index)
            else 
                ind_right = findfirst(x->x==qregion, new_index) 
            end
            push!(blocks_site, Block(2,ind_left))
        end  
        ind_left += 1
    end
    append!(blocks_mps, [blocks_site])
    return link_indices_forward, blocks_mps, max_num_qregions
end

"""
Given some constraints

lb <= A x <= ub, 

output :

- mps with uniform entries and flux by default at site 1.
- link indices that assign mps flux at first site. 
- link indices that assign mps flux at last site. 
- max number of qregions found during backward sweep.

Args : 
- flux_center: flux position 
- block_dim: block dim of each tensor 
- verbose: if true dynamically prints out site indices during backward/forward sweeps  
- ensemble_entries: ensemble distribution of entries. Default to all tensor entries constant and equal to 1 
"""

function constraints_to_mps(A::Matrix{Int}, lb::Vector{Int}, ub::Vector{Int}; flux_center=1, verbose=false, block_dim=1, ensemble_entries="constant")
    m, N = size(A)

    results_forward = Threads.@spawn constraints_to_indices(A, lb, ub; verbose=verbose)
    results_backward = Threads.@spawn constraints_to_indices(reverse(A,dims=2), lb, ub; verbose=verbose)

    link_indices_forward, blocks_forward, max_num_qregions_forward = fetch(results_forward)
    link_indices_backward, blocks_backward, max_num_qregions_backward = fetch(results_backward)

    max_num_qregions = maximum([max_num_qregions_forward, max_num_qregions_backward])

    reverse!(link_indices_backward)
    reverse!(blocks_backward)

    link_inds = Vector{Index}()
    blocks = []
    link_inds_forward = Vector{Index}()
    link_inds_backward = Vector{Index}()
    for j in 1:N-1 
        push!(link_inds_forward, Index(map(i -> link_indices_forward[j][i] => block_dim, eachindex(link_indices_forward[j]))))
        push!(link_inds_backward, Index(map(i -> link_indices_backward[j][i] => block_dim, eachindex(link_indices_backward[j]))))
    end

    for j in 1:N-1
        if j < flux_center
            push!(link_inds, link_inds_forward[j]) 
            append!(blocks, [blocks_forward[j]])
        else
            push!(link_inds, link_inds_backward[j])
            append!(blocks, [blocks_backward[j]])
        end
    end

    if flux_center == N 
        append!(blocks, [blocks_forward[N]])
    else
        append!(blocks, [blocks_backward[N]])
    end

    site_inds = Vector{Index}(undef, N)
    for j in 1:N 
        zeroqn = ints_to_QNs([0 for i in 1:m])
        nonzeroqn = ints_to_QNs([A[i,j] for i in 1:m])
        site_inds[j] = Index([zeroqn=>1, nonzeroqn=>1])
    end

    fl = QRegion([Box(lb,ub)]) # flux 

    v = Vector{ITensor}(undef, N)

    # TODO : check that it works for flux center away from ends 
    if flux_center == 1
        v[1] = randomITensor(fl, (dag(site_inds[1]), dag(link_inds[1])))
    else 
        v[1] = randITensor((dag(site_inds[1]), link_inds[1]), blocks[1])
    end
    for j in 2:N-1 
        if j < flux_center
            v[j] = randITensor((dag(site_inds[j]),dag(link_inds[j-1]),link_inds[j]), blocks[j])
        elseif j == flux_center # TODO : cannot use randITensor 
            v[j] = randomITensor(fl, (dag(site_inds[j]),dag(link_inds[j-1]),dag(link_inds[j])))
        else
            v[j] = randITensor((dag(site_inds[j]),dag(link_inds[j]),link_inds[j-1]), blocks[j]) # mind the order of indices! Compare with case j < flux_center
        end
    end

    if flux_center == N 
        v[N] = randomITensor(fl, (dag(site_inds[N]),dag(link_inds[N-1])))
    else 
        v[N] = randITensor((dag(site_inds[N]),link_inds[N-1]), blocks[N])
    end

    if ensemble_entries == "constant" # assign all entries to 1 (default)
        for i in 1:N 
            fill!(v[i].tensor.storage.data, 1.0)
        end
    elseif ensemble_entries == "uniform" # assign entries uniformly random
        for i in 1:N 
            rand!(v[i].tensor.storage.data)
        end
    end 

    mps = MPS(v)

    return mps, link_inds_backward, link_inds_forward, max_num_qregions
end

""" 
Map vector of integers to QNs. 
"""
function ints_to_QNs(v::Vector{Int64})
    m = length(v)
    q = MQNStorage(ntuple(_ -> ZeroVal, Val(maxQNs))) 
    for j in 1:m
        q[j] = QNVal("q$(j)", v[j])
    end
    return QN(QNStorage(q))
end