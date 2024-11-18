import Pkg; Pkg.activate((@__DIR__)*"../../..")
using Unzip, Statistics
using ProgressMeter
using JLD2
using MatrixProductBP, MatrixProductBP.Models

include("../../src/mpbp.jl")

include((@__DIR__)*"/../../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

β = 1.0
h = 0.0
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

function run_bp!(ps, errs, J, ds;
        A0 = reshape([10 10; 0.1 0.1], 1,1,2,2))
    m_true, r_true = equilibrium_observables(RandomRegular(k), J; β, h, maxiter=10^4, tol=1e-12)
    bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)
    m = NaN
    for a in eachindex(ds)
        d = ds[a]
        svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
        bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)
        bp.μ[1].tensor = A0
        cb = CB_BPVUMPS(bp; f=spin)
        iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
        iter ≥ maxiter && @warn "BP didn't converge"
        m = abs(only(only(cb.m[end])))
        r = only(only(alternate_correlations(spin, bp)))
        ps[a] = m
        er = max(abs(m - m_true), abs(r - r_true))
        errs[a] = er
        println("\n\n d=$d, m=$m, err=$(er)\n\n")
        if er < tol_magnetiz
            return d
        end
        A0 = bp.μ[1].tensor
        m = NaN
    end
    return typemax(Int)
end

ps = [fill(NaN, length(ds)) for _ in Js]
errs = [fill(NaN, length(ds)) for _ in Js]

A0 = reshape([0.1 0.1; 0.1 10], 1,1,2,2)

for i in eachindex(Js)
    J = Js[i]
    run_bp!(ps[i], errs[i], J, ds; A0)
    println("###\nFinished J=$J, $i of ", length(Js), "\n")
end

# function run_bp(J)
#     m_ss, = equilibrium_observables(RandomRegular(k), J; β, h)
#     bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)
#     m = Inf
#     er = Inf
#     for d in ds
#         svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
#         cb = CB_BPVUMPS(bp; f=spin)
#         iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
#         Base.GC.gc()
#         m = [only(only(m)) for m in cb.m][end]
#         er = abs(m - m_ss) / abs(m_ss)
#         if er < tol_magnetiz
#             return (m, d, er)
#         end
#     end
#     return (m, typemax(Int), er)
# end

# m = zeros(length(Js))
# d = zeros(Int, length(Js))
# errs = zeros(length(Js))
# for i in eachindex(Js)
#     J = Js[i]
#     m[i], d[i], errs[i] = run_bp(J)
#     println("###\nFinished J=$J, $i of ", length(Js), "\n")
# end

m = ps

jldsave((@__DIR__)*"/../../data/glauber_bonddims5.jld2"; Js, h, β, k, tol_magnetiz, m, ds, errs)

@telegram "Glauber bond dim"