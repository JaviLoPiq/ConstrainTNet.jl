# TODO : Refactor to ignore singular values below machine precision, 
# avoiding spurious contributions without artificially setting precision to 1e-30.
function custom_truncated_blockdim(
    S::DiagMatrix, docut::Real; singular_values=false, truncate=true, min_blockdim=0
  )
    full_dim = diaglength(S)
    (docut > 1e-30) || (docut = 1e-30)
    min_blockdim = min(min_blockdim, full_dim) 
    newdim = 0
    val = singular_values ? getdiagindex(S, newdim + 1)^2 : abs(getdiagindex(S, newdim + 1))
    while newdim + 1 ≤ full_dim && val > docut
        newdim += 1
        if newdim + 1 ≤ full_dim
            val =
            singular_values ? getdiagindex(S, newdim + 1)^2 : abs(getdiagindex(S, newdim + 1))
        end
    end
    if truncate 
        (newdim >= min_blockdim) || (newdim = min_blockdim) 
    end
    return newdim
end

function constrained_svd(T::BlockSparseMatrix{ElT}, blocks::Vector{Vector{Block{2}}}; kwargs...) where {ElT}
    alg::String = get(kwargs, :alg, "divide_and_conquer")
    min_blockdim::Int = get(kwargs, :min_blockdim, 0)
    truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)
    nnzblocksT = nnzblocks(T)
    Us = Vector{DenseTensor{ElT,2}}(undef, nnzblocksT)
    Ss = Vector{DiagTensor{real(ElT),2}}(undef, nnzblocksT)
    Vs = Vector{DenseTensor{ElT,2}}(undef, nnzblocksT)

    d = Vector{real(ElT)}()
  
    num_merged_blocks = length(blocks)
    index_merged_blocks = 1
    index_blocks = 1
    while index_merged_blocks <= num_merged_blocks
        vec_indices = Vector{Int64}() 
        num_sub_blocks = length(blocks[index_merged_blocks])
        vec_block_dims = Vector{Tuple}()
        for m in 1:num_sub_blocks
            bl = blocks[index_merged_blocks][m]
            offsetT = NDTensors.offset(T, bl)
            blockdimsT = blockdims(T, bl)
            push!(vec_block_dims, blockdimsT)
            blockdimT = prod(blockdimsT)
            append!(vec_indices, (offsetT+1): (offsetT+blockdimT))
        end
        dataTslice = @view data(storage(T))[vec_indices]

        blockdimsT = vec_block_dims[1]
        for m in 2:num_sub_blocks
            if blockdimsT[1] == vec_block_dims[m][1]
                blockdimsT = (blockdimsT[1], blockdimsT[2]+vec_block_dims[m][2])
            elseif blockdimsT[2] == vec_block_dims[m][2]
                blockdimsT = (blockdimsT[1]+vec_block_dims[m][1], blockdimsT[2])
            else
                error("Blocks must have same number of columns or rows.")
            end
        end
        blockT = tensor(Dense(dataTslice), blockdimsT)
        USVb = svd(blockT; alg=alg)
        if isnothing(USVb)
            return nothing
        end
        Ub, Sb, Vb = USVb
        if num_sub_blocks == 1
            Us[index_blocks] = Ub
            Ss[index_blocks] = Sb
            Vs[index_blocks] = Vb
            append!(d, data(Sb))
            index_blocks += 1
        else 
            if blockdimsT[1] == vec_block_dims[2][1] # compression along "row" direction
                col_ind = 1
                for n in 1:num_sub_blocks
                    num_cols = vec_block_dims[n][2]
                    Us[index_blocks] = Ub
                    Ss[index_blocks] = Sb
                    Vs[index_blocks] = Vb[col_ind:col_ind+num_cols-1,:] # note that since this is the adjoint, cols <-> rows
                    append!(d, data(Sb))
                    col_ind += num_cols
                    index_blocks += 1
                end
            else # compression along "column" direction
                row_ind = 1
                for n in 1:num_sub_blocks
                    num_rows = vec_block_dims[n][1]
                    Us[index_blocks] = Ub[row_ind:row_ind+num_rows-1,:]
                    Ss[index_blocks] = Sb
                    Vs[index_blocks] = Vb
                    append!(d, data(Sb))
                    row_ind += num_rows
                    index_blocks += 1
                end
            end
        end
        index_merged_blocks += 1
    end

    # Square the singular values to get
    # the eigenvalues
    d .= d .^ 2
    sort!(d; rev=true)
  
    # Get the list of blocks of T
    # that are not dropped
    nzblocksT = sort(nzblocks(T)) # TODO : Can we not just call blocks? Compare with ITensors version
    dropblocks = Int[]

    # TODO : Refactor truncation to avoid unnecessary block dimension increases and artificial zero 
    # singular values when shifting the canonical center.

    truncerr, docut = truncate!(d; kwargs...)
    for n in 1:nnzblocks(T)
        blockdim = custom_truncated_blockdim(
            Ss[n], docut; min_blockdim, singular_values=true, truncate
        )
        if blockdim == 0
            push!(dropblocks, n)
        else
            Strunc = tensor(Diag(storage(Ss[n])[1:blockdim]), (blockdim, blockdim))
            Us[n] = Us[n][1:NDTensors.dim(Us[n], 1), 1:blockdim]
            Ss[n] = Strunc
            Vs[n] = Vs[n][1:NDTensors.dim(Vs[n], 1), 1:blockdim]
        end
    end
    deleteat!(Us, dropblocks)
    deleteat!(Ss, dropblocks)
    deleteat!(Vs, dropblocks)
    deleteat!(nzblocksT, dropblocks)

    # The number of non-zero blocks of T remaining
    nnzblocksT = length(nzblocksT)
    #
    # Make indices of U and V 
    # that connect to S
    #
    i1 = ind(T, 1)
    i2 = ind(T, 2)
    uind = dag(sim(i1))
    vind = dag(sim(i2))
    
    resize!(uind, nnzblocksT) 
    resize!(vind, nnzblocksT)
    for (n, blockT) in enumerate(nzblocksT)
        Udim = size(Us[n], 2)
        b1 = NDTensors.block(i1, blockT[1])
        NDTensors.setblock!(uind, NDTensors.resize(b1, Udim), n)
        Vdim = size(Vs[n], 2)
        b2 = NDTensors.block(i2, blockT[2])
        NDTensors.setblock!(vind, NDTensors.resize(b2, Vdim), n)
    end
    
    #
    # Put the blocks into U,S,V
    # 
  
    nzblocksU = Vector{Block{2}}(undef, nnzblocksT)
    nzblocksS = Vector{Block{2}}(undef, nnzblocksT)
    nzblocksV = Vector{Block{2}}(undef, nnzblocksT)
  
    for (n, blockT) in enumerate(nzblocksT)
        blockU = (blockT[1], UInt(n))
        nzblocksU[n] = blockU

        blockS = (n, n)
        nzblocksS[n] = blockS

        blockV = (blockT[2], UInt(n))
        nzblocksV[n] = blockV
    end
  
    indsU = setindex(inds(T), uind, 2)
    indsV = setindex(inds(T), vind, 1)
    indsV = NDTensors.permute(indsV, (2, 1))
  
    indsS = setindex(inds(T), dag(uind), 1)
    indsS = setindex(indsS, dag(vind), 2)
    U = BlockSparseTensor(ElT, undef, nzblocksU, indsU)
    S = DiagBlockSparseTensor(real(ElT), undef, nzblocksS, indsS)
    V = BlockSparseTensor(ElT, undef, nzblocksV, indsV)
    for n in 1:nnzblocksT
        Ub, Sb, Vb = Us[n], Ss[n], Vs[n]

        blockU = nzblocksU[n]
        blockS = nzblocksS[n]
        blockV = nzblocksV[n]

        if VERSION < v"1.5"
            # In v1.3 and v1.4 of Julia, Ub has
            # a very complicated view wrapper that
            # can't be handled efficiently
            Ub = copy(Ub)
            Vb = copy(Vb)
        end

        blockview(U, blockU) .= Ub
        blockviewS = blockview(S, blockS)
        for i in 1:diaglength(Sb)
            NDTensors.setdiagindex!(blockviewS, NDTensors.getdiagindex(Sb, i), i)
        end

        blockview(V, blockV) .= Vb
    end
    truncerr = 0
    return U, S, V, Spectrum(d, truncerr)
end

# assumes that corresponding matrix T to be svd'd 
# has the merged indices (physical+one virtual) as row index and that there can't be blocks to be 
# merged along column direction 
function merge_blocks(T::BlockSparseMatrix{ElT}) where {ElT}
    blocks = sort(nzblocks(T)) # TODO : can we change nzblocks() so that blocks are always sorted?
    merged_blocks = Vector{Vector{Block{2}}}()
    push!(merged_blocks, [blocks[1]])
    for block in blocks[2:end] 
        if block[1] == merged_blocks[end][1][1]
            push!(merged_blocks[end], block)
        else
            @assert block[1] > merged_blocks[end][1][1]
            push!(merged_blocks, [block])
        end
    end
    return merged_blocks
end