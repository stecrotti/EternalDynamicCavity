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

# include("mpbp.jl")
include("vumps.jl")
include("vidal.jl")
# include("VUMPS.jl")

export tr
# export truncate_vumps, iterate_bp_vumps, belief, pair_belief
export iMPS, canonicalize, bond_dims, overlap
export iterate_bp_vumps_bipartite
export vumps, VUMPSState, resize!
export iterate_bp_vumps_mpskit, CallbackBPVUMPS

end
