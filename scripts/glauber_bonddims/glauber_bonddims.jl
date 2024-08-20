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

include((@__DIR__)*"/../../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

β = 1.0
h = 0.1
k = 3
T = 0

Js = 0.05:0.05:1.0

ds = 1:12
tol_magnetiz = 1e-3

maxiter = 70
tol = 1e-6
maxiter_vumps = 10
tol_vumps = 1e-14

spin(x,args...) = 3-2x

function run_bp(J)
    m_ss, = equilibrium_observables(RandomRegular(k), J; β, h)
    bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)
    m = Inf
    er = Inf
    for d in ds
        svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
        cb = CB_BPVUMPS(bp; f=spin)
        iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
        Base.GC.gc()
        m = [only(only(m)) for m in cb.m][end]
        er = abs(m - m_ss) / abs(m_ss)
        if er < tol_magnetiz
            return (m, d, er)
        end
    end
    return (m, typemax(Int), er)
end

m = zeros(length(Js))
d = zeros(Int, length(Js))
errs = zeros(length(Js))
for i in eachindex(Js)
    J = Js[i]
    m[i], d[i], errs[i] = run_bp(J)
    println("###\nFinished J=$J, $i of ", length(Js), "\n")
end

jldsave((@__DIR__)*"/../../data/glauber_bonddims2.jld2"; Js, h, β, k, tol_magnetiz, m, d, errs, ds)

@telegram "Glauber bond dim"