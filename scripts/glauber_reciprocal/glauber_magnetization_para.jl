using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter
using JLD2
using MatrixProductBP, MatrixProductBP.Models
using Plots

include("../../src/mpbp.jl")

include((@__DIR__)*"/../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

J = 0.4
β = 1.0
h = 0.2
k = 3
T = 0

(m_ss, r_ss) = equilibrium_observables(RandomRegular(k), J; β, h)

ds = 1:2:13

maxiter = 70
tol = 1e-7
maxiter_vumps = 10
tol_vumps = 1e-14

spin(x,args...) = 3-2x
bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)

iters, cbs = map(eachindex(ds)) do a
    only(bp.μ).tensor = reshape([0.1 0.1; 0.1 10], 1, 1, 2, 2)
    svd_trunc = TruncVUMPS(ds[a]; maxiter=maxiter_vumps, tol=tol_vumps)
    cb = CB_BPVUMPS(bp; f=spin)
    iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
end |> unzip

m_bp = [[only(only(m)) for m in cb.m] for cb in cbs]

ps = [m[end] for m in m_bp]

jldsave((@__DIR__)*"/../data/glauber_para.jld2"; J, h, β, ds, m_bp, ps, m_ss)

@telegram "vumps glauber para finished"