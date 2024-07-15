using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter
using MatrixProductBP, MatrixProductBP.Models
using JLD2
include("../src/mpbp.jl")

include((@__DIR__)*"/../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)

J = 0.4
β = 1.0
h = 0.2
k = 3

maxiter = 50
tol = 1e-15

maxiter_vumps = 100
tol_vumps = 1e-14

d = 15
svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
maxiter = 20
tol = 0

spin(x,args...) = 3-2x
bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)

function MyCB(bp)
    c = zeros(ComplexF64, 0)
    cb = CB_BP(bp; f=spin)
    function callback(bp, it, svd_trunc)
        push!(c, only(only(autocorrelations(spin, bp))))
        cb(bp, it, svd_trunc)
    end
end

cb = MyCB(bp)
iterate!(bp; maxiter, svd_trunc, tol, cb)

(m_ss, r_ss) = equilibrium_observables(RandomRegular(3), J; β, h)

using Plots
pl1 = plot([only(only(m)) for m in cb.cb.m], xlabel="iter", ylabel="magnetiz")
hline!(pl1, [m_ss, -m_ss], ls=:dash, ylims=(-1,1))

pl2 = plot(real(cb.c), xlabel="iter", ylabel="2-time autocorr")
hline!(pl2, [r_ss], ls=:dash, ylims=(-1,1))

plot(pl1, pl2)