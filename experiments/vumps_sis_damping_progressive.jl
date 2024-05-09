using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using MatrixProductBP, MatrixProductBP.Models
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter
using JLD2
using Plots

using MPSKit, KrylovKit

include((@__DIR__)*"/../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)

λ = 0.07
ρ = 0.1
d = 12
Δts = 10.0 .^ (0:-0.5:-3)
αs = pushfirst!(zeros(length(Δts)-1), 1e-2)

maxiter = 50
tol = 1e-12
maxiter_vumps = 100

A0 = reshape([0.5 0.5; 0.2 0.2], 1,1,2,2)

function foo()
    A = copy(A0)
    ε, iter, err, ovl, bel, AA = map(eachindex(Δts)) do a
        Δt = Δts[a]; α = αs[a]
        f = SISFactor(λ*Δt, ρ*Δt; α=0.0)
        A, iters, εs, errs, ovls, beliefs, As = iterate_bp_vumps_mpskit(f, d; A0=A, tol, maxiter,
            maxiter_vumps)
        println("\n####\nΔt=$Δt, $a/$(length(Δts))\n#####\n")
        εs, iters, errs, ovls, beliefs, As
    end |> unzip
end

ε, iter, err, ovl, bel, AA = foo()

εs = reduce(vcat, ε)
iters = reduce(vcat, iter)
errs = reduce(vcat, err)
ovls = reduce(vcat, ovl)
bels = reduce(vcat, bel)
ps = [b[2] for b in bels]

p_gillespie = 0.33795382376237626
x = λ / ρ
k = 3
p_ss_cme = (x*(k-1)-1) / (x*(k-1)-1 + (k-1)/k)


pl1 = plot(ps, xlabel="iter", ylabel="p(xᵢ=INFECT)")
vline!(pl1, cumsum(iters), label="", ls=:dash)
hline!(pl1, [p_gillespie], label="gillespie")
hline!(pl1, [p_ss_cme], label="CME")

@telegram "Vumps SIS damping progressive"