using MPSKit, TensorKit

function truncate_vumps(A::InfiniteUniformTensorTrain{F,3}, d::Integer) where F<:Number
    tensor = A.tensor
    t = complex.(permutedims(tensor, (1,3,2)))
    M = size(A, 1)
    q = size(A, 3)
    @assert size(A, 2) == M
    d = TensorMap(t ,(ℝ^M ⊗ ℝ^q), ℝ^M) # the same but as a type digestible by MPSKit.jl
    ψ₀ = InfiniteMPS([d])

    I = DenseMPO([MPSKit.add_util_leg(id(storagetype(MPSKit.site_type(ψ₀)), physicalspace(ψ₀, i))) for i in 1:length(ψ₀)])  # whatever
    ψ = InfiniteMPS(ℝ^q, ℝ^d) # approximate by MPS of bond dimension 5

    alg = VUMPS(; tol_galerkin=1e-10, maxiter=100) # variational approximation algorithm

    ψ, = approximate(ψ, (I, ψ₀), alg) 
    AL_resh = only(ψ.AL).data
    AL = real(reshape(AL_resh, (q, d, q)))
    Anew = InfiniteUniformTensorTrain(permutedims(AL, (1,3,2)))
    return Anew
end