import Base.∩, Base.-, Base.merge, Base.isless, Base.==

struct Box
    min_corner::Vector{Int} # Represents the minimum corner in N-dimensional space
    max_corner::Vector{Int} # Represents the maximum corner in N-dimensional space

    function Box(min_corner::Vector{Int}, max_corner::Vector{Int})
        @assert length(min_corner) == length(max_corner) "Dimensions of min_corner and max_corner must match"
        @assert all(min_corner .<= max_corner) "All elements of max_corner must be greater than or equal to min_corner"
        return new(min_corner, max_corner)
    end
end

function Base.isequal(box1::Box, box2::Box)::Bool
    all(box1.min_corner .== box2.min_corner) && all(box1.max_corner .== box2.max_corner)
end

(box1::Box == box2::Box) = isequal(box1, box2)

@inline function intersection(box1::Box, box2::Box)::Union{Box, Nothing}
    L1 = length(box1.min_corner)
    L2 = length(box2.min_corner)
    # Ensure the boxes are in the same dimension
    @assert L1 == L2 "Boxes must be in the same dimension"

    min_corner = Vector{Int}()
    max_corner = Vector{Int}()
    # Compute the intersection
    for i in 1:L1
        min_corner_coord = max(box1.min_corner[i], box2.min_corner[i])
        max_corner_coord = min(box1.max_corner[i], box2.max_corner[i])
        if min_corner_coord > max_corner_coord
            return nothing 
        else
            push!(min_corner, min_corner_coord)
            push!(max_corner, max_corner_coord)
        end
    end

    return Box(min_corner, max_corner)
end

(box1::Box ∩ box2::Box) = intersection(box1, box2)

function difference(box1::Box, box2::Box)::Vector{Box}
    inter = intersection(box1, box2) 
    
    if inter === nothing 
        return [box1]
    end

    dim_box = length(box1.min_corner)

    differences = Vector{Box}()
    for i in 1:dim_box
        if box1.min_corner[i] < inter.min_corner[i]
            temp_max_corner = copy(box1.max_corner)
            temp_max_corner[i] = inter.min_corner[i] - 1
            push!(differences, Box(copy(box1.min_corner), temp_max_corner))
        end
        if box1.max_corner[i] > inter.max_corner[i]
            temp_min_corner = copy(box1.min_corner)
            temp_min_corner[i] = inter.max_corner[i] + 1 
            push!(differences, Box(temp_min_corner, copy(box1.max_corner)))
        end
    end
    merged_differences = merge_overlapping_boxes(differences, dim_box)

    return merged_differences
end

(box1::Box - box2::Box) = difference(box1, box2)

function merge_overlapping_boxes(boxes::Vector{Box}, dim_box::Int)::Vector{Box}
    # num_boxes should be at most n, with n the dimensionality 
    num_boxes = length(boxes)
    for i in 1:num_boxes-1
        for j in i+1:num_boxes
            inter = intersection(boxes[i], boxes[j])
            if inter !== nothing 
                box = boxes[i]
                for k in 1:dim_box
                    (box.min_corner[k] < inter.min_corner[k]) && (boxes[i].max_corner[k] = inter.min_corner[k] - 1)
                    (box.max_corner[k] > inter.max_corner[k]) && (boxes[i].min_corner[k] = inter.max_corner[k] + 1)
                end
            end
        end
    end
    return boxes
end

function isless(box1::Box, box2::Box)::Bool
    for i in 1:length(box1.min_corner)
        if box1.min_corner[i] < box2.min_corner[i]
            return true
        elseif box1.min_corner[i] > box2.min_corner[i]
            return false
        end
    end
    # If all dimensions are equal, consider box1 not less than box2
    return false
end

function addition(box1::Box, box2::Box)::Box 
    return Box(box1.min_corner+box2.min_corner, box1.max_corner+box2.max_corner)
end

(box1::Box + box2::Box) = addition(box1,box2)

function addition(box::Box, int_vector::Vector{Int})::Box
    return Box(box.min_corner+int_vector, box.max_corner+int_vector)
end

(box::Box + int_vector::Vector{Int}) = addition(box,int_vector)
(int_vector::Vector{Int} + box::Box) = addition(box,int_vector)
(box::Box - int_vector::Vector{Int}) = addition(box,-int_vector)
(int_vector::Vector{Int} - box::Box) = addition(box,-int_vector)

function -(box::Box) #TODO: need this for assigning Arrow (think of better way of assigning meaning to Arrow for Box/QRegion)
    return Box(box.min_corner, box.max_corner)
end

function contained(box1::Box, box2::Box)::Bool
    for i in 1:length(box1.min_corner)
        # box1's min_corner must be >= box2's min_corner in every dimension
        if box1.min_corner[i] < box2.min_corner[i]
            return false
        end
        # box1's max_corner must be <= box2's max_corner in every dimension
        if box1.max_corner[i] > box2.max_corner[i]
            return false
        end
    end
    return true
end

(box1::Box ⊂ box2::Box) = contained(box1,box2)

# TODO : length of a QN type is fixed to MaxQNs. It would be nice to have a function val that when called on a QN returns all the nonempty entries (corresponding 
# to the number of constraints) 
function addition(box::Box, qn::QN)::Box
    if qn == QN() 
        return box 
    else 
        return Box(box.min_corner .+ [qn[i].val for i in 1:length(box.min_corner)], box.max_corner .+ [qn[i].val for i in 1:length(box.max_corner)])
    end
end

(box::Box + qn::QN) = addition(box,qn)
(qn::QN + box::Box) = addition(box,qn)
(box::Box - qn::QN) = addition(box,-qn)
(qn::QN - box::Box) = addition(-box,qn)

function multiplication(dir::Arrow, box::Box)::Box
    if Int(dir) == 1
        return box 
    else # Int(dir) == -1 
        return -box
    end
end

(dir::Arrow * box::Box) = multiplication(dir, box)

(box::Box * dir::Arrow) = (dir * box)

function Base.show(io::IO, boxes::Vector{Box})
    print(io, '[')
    for (i, box) in enumerate(boxes)
        if i > 1
            print(io, ", ")
        end
        print(io, "Box($(box.min_corner), $(box.max_corner))")
    end
    print(io, ']')
end

struct QRegion <: QNRegion
    boxes::Vector{Box}
end

function intersection(qregion1::QRegion, qregion2::QRegion)::Union{Nothing,QRegion}
    set1 = qregion1.boxes 
    set2 = qregion2.boxes 
    qregion = Vector{Box}(undef, length(set1)*length(set2))
    count = 0
    for i in 1:length(set1)
        for j in 1:length(set2)
            inter = set1[i] ∩ set2[j] 
            if inter !== nothing    
                count += 1
                qregion[count] = inter
            end
        end 
    end 
    resize!(qregion, count)
    if qregion != []
        return QRegion(qregion)
    else
        return nothing 
    end
end

(qregion1::QRegion ∩ qregion2::QRegion) = intersection(qregion1, qregion2)

function is_intersect(qregion1::QRegion, qregion2::QRegion)::Bool 
    for i in eachindex(qregion1.boxes)
        for j in eachindex(qregion2.boxes)
            inter = qregion1.boxes[i] ∩ qregion2.boxes[j] 
            if inter !== nothing 
                return true 
            end
        end 
    end 
    return false
end

function difference(qregion1::QRegion, qregion2::QRegion)::QRegion #TODO: make cleaner? improve by ordering (N^2 -> N log N)
    for i in eachindex(qregion2.boxes)
        boxes1_new = Vector{Box}()
        for j in eachindex(qregion1.boxes)
            box1 = qregion1.boxes[j] - qregion2.boxes[i]
            append!(boxes1_new, box1)
        end 
        qregion1 = QRegion(boxes1_new)
    end 
    return qregion1
end

(qregion1::QRegion - qregion2::QRegion) = difference(qregion1, qregion2)

#TODO: the resulting QRegion will have overlapping boxes. This should be fine if we only ever 
# care about whether the resulting QRegion never becomes part of an Index 
# e.g. we will only need this for checking flux condition
function addition(qregion1::QRegion, qregion2::QRegion)::QRegion 
    qtot = Vector{Box}()
    for i in 1:length(qregion1.boxes)
        for j in 1:length(qregion2.boxes)
            box1 = qregion1.boxes[i]
            box2 = qregion2.boxes[j]
            push!(qtot, box1+box2)
        end
    end
    return QRegion(qtot)
end

(qregion1::QRegion + qregion2::QRegion) = addition(qregion1,qregion2)

#TODO: find a way to find ordering of QRegion; ITensor by default orders QNs when doing contractions (e.g. merging of QNs) for speedup. See e.g. combineblocks that relies on ordered QNs
#Currently the ordering is based on the first box alone of each qregion
#TODO: construct QRegions so that they are always ordered that way we don't need to call sort! here
function isless(qregion1::QRegion, qregion2::QRegion)::Bool 
    QRegion(sort!(qregion1.boxes))
    QRegion(sort!(qregion2.boxes))
    if isless(qregion1.boxes[1], qregion2.boxes[1])
        return true 
    end 
    return false
end
  
function -(qregion::QRegion)
    boxes = Vector{Box}() 
    for i in eachindex(qregion.boxes)
        push!(boxes, -qregion.boxes[i])
    end 
    return QRegion(boxes)
end

function addition(qregion::QRegion, int_vector::Vector{Int})::QRegion
    new_box_vec = Vector{Box}() 
    for i in eachindex(qregion.boxes)
        push!(new_box_vec, qregion.boxes[i]+int_vector)
    end
    return QRegion(new_box_vec)
end

(qregion::QRegion + int_vector::Vector{Int}) = addition(qregion, int_vector)
(int_vector::Vector{Int} + qregion::QRegion) = addition(qregion, int_vector)
(qregion::QRegion - int_vector::Vector{Int}) = addition(qregion, -int_vector)
(int_vector::Vector{Int} - qregion::QRegion) = addition(-qregion, int_vector)

function addition(qregion::QRegion, qn::QN)::QRegion
    new_box_vec = Vector{Box}() 
    for i in eachindex(qregion.boxes)
        push!(new_box_vec, qregion.boxes[i]+qn)
    end
    return QRegion(new_box_vec)
end

#TODO: will we ever need negative of Box/QRegion?
(qregion::QRegion + qn::QN) = addition(qregion, qn)
(qn::QN + qregion::QRegion) = addition(qregion, qn)
(qregion::QRegion - qn::QN) = addition(qregion, -qn)
(qn::QN - qregion::QRegion) = addition(-qregion, qn)

#TODO: better naming? same for other interior 
function interior(qn::QN, qregion::QRegion)::Union{Nothing,QRegion}
    dim_qn = length(qregion.boxes[1].min_corner)
    for j in eachindex(qregion.boxes)
        if all(qregion.boxes[j].min_corner .<= [qn[i].val for i in 1:dim_qn] .<= qregion.boxes[j].max_corner)
            return qregion
        end 
    end
end

function contained(qn::QN, qregion::QRegion)::Bool
    dim_qn = length(qregion.boxes[1].min_corner)
    for j in eachindex(qregion.boxes)
        if all(qregion.boxes[j].min_corner .<= [qn[i].val for i in 1:dim_qn] .<= qregion.boxes[j].max_corner)
            return true 
        end 
    end
    return false 
end

(qn::QN ⊂ qregion::QRegion) = contained(qn, qregion)

# TODO : think what would happen if we have e.g. qregion1 = QRegion(0,0,0,1) and qregion2 = QRegion[(0,0,0,0),(0,1,0,1),(0,2,0,2)],
# clearly qregion1 ⊂ qregion2. Note that the format of qregion2 may appear from the difference of two 
# boxes as the merging is always along the x direction and this may leave union of vertical stuff
function contained(qregion1::QRegion, qregion2::QRegion)::Bool 
    for i in 1:length(qregion1.boxes)
        for j in 1:length(qregion2.boxes)
            box1 = qregion1.boxes[i]
            box2 = qregion2.boxes[j]
            if box1 ⊂ box2 
                break
            end
            if j == length(qregion2.boxes)
                return false 
            end
        end 
    end
    return true 
end

(qregion1::QRegion ⊂ qregion2::QRegion) = contained(qregion1,qregion2)

#TODO: does it make sense to consider Arrow for QRegion?
function (dir::Arrow * qregion::QRegion)::QRegion
    mqregion = Vector{Box}(undef, length(qregion.boxes))
    for i in 1:length(mqregion)
      mqregion[i] = dir * qregion.boxes[i]
    end
    return QRegion(mqregion)
end
  
(qregion::QRegion * dir::Arrow) = (dir * qregion)

function difference(qregion1_vec::Vector{QRegion}, qregion2_vec::Vector{QRegion})::Vector{QRegion} #TODO: improve by ordering (N^2 -> N log N) 
    qregion_vec_new = Vector{QRegion}()
    qregion1_vec_new = copy(qregion1_vec)
    qregion2_vec_new = copy(qregion2_vec)
    for i in 1:length(qregion1_vec_new)
        for j in 1:length(qregion2_vec)
            qregion1_vec_new[i] = qregion1_vec_new[i] - qregion2_vec[j]
        end 
    end
    for i in 1:length(qregion2_vec_new)
        for j in 1:length(qregion1_vec)
            qregion2_vec_new[i] = qregion2_vec_new[i] - qregion1_vec[j]
        end 
    end
    for i in eachindex(qregion1_vec_new)
        append!(qregion_vec_new, [qregion1_vec_new[i]])
    end
    for i in eachindex(qregion2_vec_new)
        append!(qregion_vec_new, [qregion2_vec_new[i]])
    end
    return qregion_vec_new 
end

(qregion1_vec::Vector{QRegion} - qregion2_vec::Vector{QRegion}) = difference(qregion1_vec, qregion2_vec)

function intersection(qregion1_vec::Vector{QRegion}, qregion2_vec::Vector{QRegion})::Union{Nothing,Vector{QRegion}}
    qregion_vec = Vector{QRegion}()
    for i in eachindex(qregion1_vec)
        for j in eachindex(qregion2_vec)
            inter = qregion1_vec[i] ∩ qregion2_vec[j] 
            if inter !== nothing
                append!(qregion_vec, [inter]) 
            end
        end 
    end 
    if qregion_vec != []
        return qregion_vec
    else
        return nothing
    end
end

(qregion1_vec::Vector{QRegion} ∩ qregion2_vec::Vector{QRegion}) = intersection(qregion1_vec, qregion2_vec)

function addition(qregion_vec::Vector{QRegion}, int_vector::Vector{Int})::Vector{QRegion}
    new_qregion_vec = Vector{QRegion}() 
    for i in eachindex(qregion_vec)
        push!(new_qregion_vec, qregion_vec[i]+int_vector)
    end
    return new_qregion_vec
end

(qregion_vec::Vector{QRegion} + int_vector::Vector{Int}) = addition(qregion_vec, int_vector)
(int_vector::Vector{Int} + qregion_vec::Vector{QRegion}) = addition(qregion_vec, int_vector)
(qregion_vec::Vector{QRegion} - int_vector::Vector{Int}) = addition(qregion_vec, -int_vector)
(int_vector::Vector{Int} - qregion_vec::Vector{QRegion}) = addition(qregion_vec, -int_vector)

function Base.hash(qregion::QRegion, h::UInt)
    hash(qregion.boxes, h)
end

Base.isequal(qregion1::QRegion, qregion2::QRegion) = isequal(qregion1.boxes, qregion2.boxes)

(qregion1::QRegion == qregion2::QRegion) = isequal(qregion1, qregion2)

function interior(int_vector::Vector{Int}, qregion_vector::Vector{QRegion})::Union{Nothing,QRegion} #TODO: int_vector -> Vector{QN}? Will this be ever used?
    for i in eachindex(qregion_vector)
        qregion = qregion_vector[i]
        for j in eachindex(qregion.boxes)
            if all(qregion.boxes[j].min_corner .<= int_vector .<= qregion.boxes[j].max_corner)
                return qregion
            end 
        end
    end
    return nothing
end

function contained(int_vector::Vector{Int}, qregion_vector::Vector{QRegion})::Bool #TODO: int_vector -> Vector{QN}? Will this be ever used?
    for i in eachindex(qregion_vector)
        qregion = qregion_vector[i]
        for j in eachindex(qregion.boxes)
            if all(qregion.boxes[j].min_corner .<= int_vector .<= qregion.boxes[j].max_corner)
                return true
            end 
        end
    end
    return false
end

(int_vector::Vector{Int} ⊂ qregion_vector::Vector{QRegion}) = contained(int_vector, qregion_vector)

function interior(qn::QN, qregion_vector::Vector{QRegion})::Union{Nothing,QRegion}
    for i in eachindex(qregion_vector)
        qregion = qregion_vector[i]
        qregion_new = interior(qn, qregion)
        if qregion_new !== nothing  
            return qregion_new 
        end
    end
    return nothing
end

function contained(qn::QN, qregion_vector::Vector{QRegion})::Bool
    for i in eachindex(qregion_vector)
        qregion = qregion_vector[i]
        if qn ⊂ qregion 
            return true 
        end
    end
    return false 
end

(qn::QN ⊂ qregion_vector::Vector{QRegion}) = contained(qn, qregion_vector)

function interior(qregion_sub::QRegion, qregion_vector::Vector{QRegion})::Union{Nothing,QRegion}
    for i in eachindex(qregion_vector)
        qregion = qregion_vector[i]
        if qregion_sub ⊂ qregion 
            return qregion 
        end
    end
    return nothing
end

function contained(qregion_sub::QRegion, qregion_vector::Vector{QRegion})::Union{Nothing,QRegion}
    for i in eachindex(qregion_vector)
        qregion = qregion_vector[i]
        if qregion_sub ⊂ qregion 
            return true  
        end
    end
    return false 
end

(qregion_sub::QRegion ⊂ qregion_vector::Vector{QRegion}) = contained(qregion_sub, qregion_vector)

function (∩)(::Any, ::Nothing)::Nothing
    return nothing
end

function (∩)(::Nothing, ::Any)::Nothing
    return nothing
end

function (-)(A::Any, ::Nothing)
    return A
end

function (-)(::Nothing, A::Any)
    return -A
end

function Base.show(io::IO, qregion::QRegion)
    print(io, "QRegion([")
    for (i, box) in enumerate(qregion.boxes)
        if i > 1
            print(io, ", ")
        end
        print(io, "Box($(box.min_corner), $(box.max_corner))")
    end
    print(io, "])")
end

function Base.show(io::IO, qregs::Vector{QRegion})
    print(io, '[')
    for (i, qreg) in enumerate(qregs)
        if i > 1
            print(io, ", ")
        end
        print(io, qreg)
    end
    print(io, ']')
end
