"""
    Fast initialization of tensor components from blocks; with uniformly random entries (in (0,1))
""" 
function randITensor(inds::Indices, blocks::Vector{Block{N}}) where {N} 
    is = Tuple(inds)
    @show is
    T = BlockSparseTensor(Float64, undef, blocks, is)
    rand!(T.storage.data)
    return itensor(T)
end

function constrained_factorize(A::ITensor, Linds...; kwargs...)
    utags::TagSet = get(kwargs, :lefttags, get(kwargs, :utags, "Link,u"))
    vtags::TagSet = get(kwargs, :righttags, get(kwargs, :vtags, "Link,v"))
  
  
    Lis = commoninds(A, ITensors.indices(Linds...))
    Ris = uniqueinds(A, Lis)
    Lis_original = Lis
    Ris_original = Ris
    if isempty(Lis_original)
        α = trivial_index(Ris)
        vLα = onehot(eltype(A), α => 1)
        A *= vLα
        Lis = [α]
    end
    if isempty(Ris_original)
        α = trivial_index(Lis)
        vRα = onehot(eltype(A), α => 1)
        A *= vRα
        Ris = [α]
    end
    if haskey(kwargs,:new_index)
        new_index = kwargs[:new_index]
        CL = combiner_compare(Lis, new_index...)
    else
        CL = combiner(Lis...)
    end
    CR = combiner(Ris...)

    AC = A * CR * CL
  
    cL = combinedind(CL)
    cR = combinedind(CR)
  
    if inds(AC) != (cL, cR)
        AC = permute(AC, cL, cR)
    end

    USVT = constrained_svd(tensor(AC), merge_blocks(tensor(AC)); kwargs...)
    if isnothing(USVT)
        return nothing
    end
    UT, ST, VT, spec = USVT
    UC, S, VC = itensor(UT), itensor(ST), itensor(VT)
    u = commonind(S, UC)
    v = commonind(S, VC)
    U = UC * dag(CL)
    V = VC * dag(CR)

    settags!(U, utags, u)
    settags!(S, utags, u)
    settags!(S, vtags, v)
    settags!(V, vtags, v)
  
    u = settags(u, utags)
    v = settags(v, vtags)
  
    if isempty(Lis_original)
        U *= dag(vLα)
    end
    if isempty(Ris_original)
        V *= dag(vRα)
    end

    if length(inds(U)) == 3
        projected_ind = inds(U)[3]
    else
        projected_ind = inds(U)[2]
    end
    qregions = Vector{Pair{QRegion, Int64}}()
    for qregion in projected_ind.space 
        if !(qregion ∈ qregions)
            push!(qregions, qregion)
        end
    end
    new_ind = Index(qregions)
    U *= constrained_delta(dag(projected_ind), new_ind)
    SV = S*V*constrained_delta(projected_ind, dag(new_ind); unique=false)
    return U, SV
end

function combiner_compare(is1::Indices, is2::Index; kwargs...)
    tags = get(kwargs, :tags, "CMB,Link")
    vec_qnregions_2 = [qn(is2, i) for i in 1:nblocks(is2)]
    set_qnregions_1 = Set{QRegion}()
    vec_qnregions_1 = Vector{QRegion}()
    block_dims_1 = Vector{Int}()
    dim_box = length(vec_qnregions_2[1].boxes[1].min_corner) # Number of constraints

    if length(is1) == 1  # Handle boundary terms
        for i in 1:nblocks(is1[1])
            qn1 = qn(is1[1], i)
            new_qn = interior(qn1, vec_qnregions_2)
            if new_qn !== nothing
                push!(set_qnregions_1, new_qn)
                push!(vec_qnregions_1, new_qn)
                push!(block_dims_1, blockdim(is1[1], i))
            else
                # Add as minimal bounding box
                push!(vec_qnregions_1, QRegion([Box([qn1[j].val for j in 1:length(dim_box)], [qn1[j].val for j in 1:length(dim_box)])]))
                push!(block_dims_1, blockdim(is1[1], i))
            end
        end
    elseif length(is1) == 2  # Handle bulk terms
        for j in 1:nblocks(is1[2])
            for i in 1:nblocks(is1[1])
                qni = qn(is1[1], i)
                qnj = qn(is1[2], j)
                qn1 = qni + qnj
                qregion_new = interior(qn1, vec_qnregions_2)
                if qregion_new !== nothing
                    push!(set_qnregions_1, qregion_new)
                    push!(vec_qnregions_1, qregion_new)
                    push!(block_dims_1, blockdim(is1[1], i) * blockdim(is1[2], j))
                else
                    push!(vec_qnregions_1, qn1)
                    push!(block_dims_1, blockdim(is1[1], i) * blockdim(is1[2], j))
                end
            end
        end
    else
        error("Can only combine QNRegions for rank-2 or rank-3 tensors.")
    end

    new_ind = Index([vec_qnregions_1[i] => block_dims_1[i] for i in 1:length(vec_qnregions_1)], tags)
    comb_ind, perm, comb = ITensors.combineblocks(new_ind)
    return itensor(Combiner(perm, comb), (dag(comb_ind), dag.(is1)...))
end

  """
  Merge repeated QRegions from index (ind1) to a similar index (ind2) with non-repeated QRegions, 
  using a projection method like the delta function. Set unique to true to project only one QRegion 
  in cases where both QRegions and blocks are repeated, necessary for SVD with constrained tensors 
  where U has repeated elements. Set unique to false when dealing with repeated QRegions but distinct blocks, 
  as in S*V.
"""
function constrained_delta(ind1::Index, ind2::Index; unique::Bool=true)
    is = (ind1, ind2)
    qregions1 = ind1.space # inds[1] corresponds to index w/ repeated qregions 
    qregions2 = ind2.space
    blocks = Vector{Block{2}}()
    if unique 
        qregions1_unique = []
        for i in eachindex(qregions1)
            if !(qregions1[i] ∈ qregions1_unique)
                push!(qregions1_unique, qregions1[i])
                j = findfirst(x -> x == qregions1[i], qregions2)
                push!(blocks,Block(i,j)) 
            end
        end
    else
        for i in eachindex(qregions1)
            j = findfirst(x -> x == qregions1[i], qregions2)
            push!(blocks,Block(i,j)) 
        end
    end

    T = DiagBlockSparseTensor(one(Float64), blocks, is)
    return itensor(T)
end  