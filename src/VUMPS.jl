using LinearAlgebra
using TensorTrains, TensorTrains.UniformTensorTrains
using MPSExperiments
using MatrixFactorizations
using TensorKit, MPSKit
using Tullio
using Random
using ProgressMeter
using Unzip
using Statistics
using MPSExperiments


using Logging
Logging.disable_logging(Logging.Info)

# A = rand(50, 50, 4)
using JLD2
A = load((@__DIR__)*"/../tmp.jld2")["A"]
q = size(A, 3)
m = size(A, 1)
ψ = InfiniteMPS([TensorMap(A, (ℝ^m ⊗ ℝ^q), ℝ^m)])
p = InfiniteUniformTensorTrain(A)

maxiter_vumps = 200
maxiter_ortho = 10
maxiter_fixedpoint = 10
tol = 1e-12

d = 6
δs = fill(NaN, maxiter_vumps+1)
# AL, AR = vumps_original(A, d; maxiter, δs)
AL, AR = vumps(A, d; maxiter_vumps, maxiter_ortho, maxiter_fixedpoint, tol, δs)

pL = InfiniteUniformTensorTrain(AL)
ovlL = abs(1 - dot(p, pL))
pR = InfiniteUniformTensorTrain(AR)
ovlR = abs(1 - dot(p, pR))
err_marg = max(
    maximum(abs, real(marginals(pL))[1] - real(marginals(p))[1]),
    maximum(abs, real(marginals(pR))[1] - real(marginals(p))[1])
)
@show ovlL, ovlR
Aperm = permutedims(A, (1,3,2))
B, = truncate_vumps(Aperm, d)
pp = InfiniteUniformTensorTrain(permutedims(B, (1,3,2)))
[real(marginals(p))[1] real(marginals(pL))[1] real(marginals(pR))[1] real(marginals(pp))[1]]

# ds = 4:8
# nsamples = 50
# maxiter = 10^2

# errs_marg, ovls = map(ds) do d
#     e, o = map(1:nsamples) do _
#         AL, AR = vumps(A, d; maxiter)
#         pL = InfiniteUniformTensorTrain(AL)
#         ovlL = abs(1 - dot(p, pL))
#         pR = InfiniteUniformTensorTrain(AR)
#         ovlR = abs(1 - dot(p, pR))
#         err_marg = max(
#             maximum(abs, real(marginals(pL))[1] - real(marginals(p))[1]),
#             maximum(abs, real(marginals(pR))[1] - real(marginals(p))[1])
#         )
#         ovl = max(abs(1 - dot(pL, p)), abs(1 - dot(pR, p))) 
#         err_marg, ovl
#     end |> unzip
#     mean(e), mean(o)
# end |> unzip

# using Plots
# pl_marg = plot(ds, errs_marg, label="error on marginals", m=:o, xlabel="bond dim")
# pl_ovl = plot(ds, ovls, label="1 - ovl", m=:o, xlabel="bond dim")
# plot(pl_marg, pl_ovl, legend=:bottomleft, layout=(2,1), size=(400,600), margin=10Plots.mm, yaxis=:log10)

# function one_vumps_iter!(A, l, r, AL, AR, L, R;
    #         maxiter_ortho=10, maxiter_fixedpoint=10)
    #     # bring to mixed canonical gauge
    #     ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical!(l, r, A; maxiter_ortho)
    
    #     # fixed point of mixed transfer operators
    #     left_fixedpoint!(L, ALtilde, AL; maxiter=maxiter_fixedpoint)
    #     right_fixedpoint!(R, ARtilde, AR; maxiter=maxiter_fixedpoint)
    
    #     # compute AC
    #     AC = copy(AL)
    #     for x in axes(A,3)
    #         @views AC[:,:,x] .= L * ACtilde[:,:,x] * R
    #     end
    #     # compute C
    #     C = L * Ctilde * R
    #     # compute minAcC
    #     AL, AR = minAcC(AC, C)
    
    #     return AL, AR, AC, C
    # end