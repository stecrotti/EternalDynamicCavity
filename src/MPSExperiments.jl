module MPSExperiments

using TensorTrains
using TensorTrains.UniformTensorTrains
using Tullio
using TensorCast
using Random
using LinearAlgebra
using Lazy: @forward
using ProgressMeter
using KrylovKit
using MPSKit
using TensorKit

using TensorTrains: _reshape1, _reshapeas

function pair_belief(A, B=A)
    @cast C[(aᵗ,bᵗ),(aᵗ⁺¹,bᵗ⁺¹),xᵢᵗ,xⱼᵗ] := A[aᵗ,aᵗ⁺¹,xᵢᵗ, xⱼᵗ] * B[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ]
    q = TensorTrains.UniformTensorTrains.InfiniteUniformTensorTrain(C)
    return marginals(q) |> only |> real
end
function belief(A; bij = pair_belief(A))
    sum(bij, dims=2) |> vec
end

include("vumps.jl")
include("vidal.jl")
# include("VUMPS.jl")

export tr
export truncate_vumps, iterate_bp_vumps, belief, pair_belief
export iMPS, canonicalize, bond_dims, overlap
export iterate_bp_vumps_bipartite
export vumps, VUMPSState, resize!
# export transfer_operator, infinite_transfer_operator

end
