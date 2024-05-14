module ConstrainTNet

#include("ITensors.jl/src/ITensors.jl")
#using .ITensors

using NDTensors
using NDTensors: setblock!, Block, Blocks, DiagMatrix, getdiagindex, BlockSparseMatrix
using ITensors
using ITensors: AbstractMPS, leftlim, rightlim, setrightlim!, setleftlim!, Indices, orthocenter, QNRegion, interior, maxQNs, ZeroVal, Box, 
  QRegion
using Random

include("constraint_handling.jl")
include("constrained_tensor.jl")
include("constrained_mps.jl")
include("constrained_blocks.jl")
include("constrained_optimization.jl")

# ITensors exports 
export Box, QRegion, sample!, normalize!
# ConstrainTNet exports 
export constraints_to_mps, constrained_orthogonalize!, optimizer, quadratic_cost_function
# TODO : add exports

end