# TODO : remove haskey, and force to pass in the left/right canonical indices 
function constrained_orthogonalize!(M::AbstractMPS, j::Int; verbose::Bool=false, kwargs...)
    while leftlim(M) < (j - 1)
        (leftlim(M) < 0) && setleftlim!(M, 0)
        b = leftlim(M) + 1
        linds = uniqueinds(M[b], M[b + 1]) 
        lb = linkind(M, b)
        if !isnothing(lb)
            ltags = tags(lb)
        else
            ltags = TagSet("Link,l=$b")
        end
        if haskey(kwargs, :left_canonical_indices) && haskey(kwargs, :flux_mps) 
            left_canonical_indices = kwargs[:left_canonical_indices]
            L, R = constrained_factorize(M[b], linds; tags=ltags, new_index=[left_canonical_indices[b]], kwargs...) 
            if verbose; @show b, norm(L * R - M[b]), norm(M[b]), maxlinkdim(M); end
        else
            L, R = factorize(M[b], linds; tags=ltags, kwargs...)
        end
        M[b] = L
        M[b + 1] *= R
        setleftlim!(M, b)
        if rightlim(M) < leftlim(M) + 2
            setrightlim!(M, leftlim(M) + 2)
        end
    end
  
    N = length(M)
  
    while rightlim(M) > (j + 1)
        (rightlim(M) > (N + 1)) && setrightlim!(M, N + 1)
        b = rightlim(M) - 2
        rinds = uniqueinds(M[b + 1], M[b]) 
        lb = linkind(M, b)
        if !isnothing(lb)
            ltags = tags(lb)
        else
            ltags = TagSet("Link,l=$b")
        end
        if haskey(kwargs, :right_canonical_indices) && haskey(kwargs, :flux_mps)
            right_canonical_indices = kwargs[:right_canonical_indices]
            L, R = constrained_factorize(M[b + 1], rinds; tags=ltags, new_index=[right_canonical_indices[b]], kwargs...)
            if verbose; @show b, norm(L * R - M[b + 1]), maxlinkdim(M); end
        else
            L, R = factorize(M[b + 1], rinds; tags=ltags, kwargs...)
        end
        M[b + 1] = L
        M[b] *= R
    
        setrightlim!(M, b + 1)
        if leftlim(M) > rightlim(M) - 2
            setleftlim!(M, rightlim(M) - 2)
        end
    end
    return M
end