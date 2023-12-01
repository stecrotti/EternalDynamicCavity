# import Pkg
# Pkg.activate("EmptyProject")
# Pkg.add(name="MPSKit", version="0.10.1")
# Pkg.add("TensorKit")
# Pkg.develop(path="../.julia/dev/TensorTrains/")

using TensorTrains, LinearAlgebra, Random
import TensorTrains: InfiniteUniformTensorTrain, one_normalization, _eigen
using KrylovKit

rng = MersenneTwister(1)
tensor = rand(rng, 10,10,2,3)
A = InfiniteUniformTensorTrain(tensor)
λ_, = _eigen(A)
A.tensor ./= λ_
B = one_normalization(A)
λ, l, r = _eigen(A; B)

function _eigen_new(A::InfiniteUniformTensorTrain; B = one_normalization(A))
    d, R, _ = eigsolve(B)
    _, L, _ = eigsolve(transpose(B))
    r = R[1]
    l = L[1]
    l ./= dot(l, r)
    λ = d[1]
    λ, l, r
end

λ, l, r = _eigen_new(A) 

T = 10
@assert B^T ≈ λ^T * r * l'

using TensorKit, MPSKit

m = 15

A = complex.(rand(m, 2, m))     # matrix to be truncated. size is mxm, variable with 2 states
t = TensorMap(A,(ℝ^m ⊗ ℝ^2), ℝ^m) # the same but as a type digestible by MPSKit.jl
ψ₀ = InfiniteMPS([t])

I = DenseMPO([MPSKit.add_util_leg(id(storagetype(MPSKit.site_type(ψ₀)), physicalspace(ψ₀, i))) for i in 1:length(ψ₀)])  # whatever

d = m
ψ = InfiniteMPS(ℝ^2, ℝ^d) # approximate by MPS of bond dimension d

alg = VUMPS(; tol_galerkin=1e-12, maxiter=100) # variational approximation algorithm

ψ, = approximate(ψ, (I, ψ₀), alg)   # do the truncation

println("⟨ψ,ψ₀⟩ = ", abs(dot(ψ, ψ₀)))  # compute overlap after truncation

B_ = reshape(only(ψ.AL).data, d, 2, d)
q = TensorTrains.InfiniteUniformTensorTrain(permutedims(B_, (1,3,2)))

A_ = reshape(A, m, 2, m)
p = TensorTrains.InfiniteUniformTensorTrain(permutedims(A_, (1,3,2)))

norm(only(marginals(p)) - only(marginals(q)))