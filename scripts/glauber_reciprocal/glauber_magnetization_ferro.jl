# using MPSExperiments
using Unzip, Statistics
using ProgressMeter
using JLD2
using MatrixProductBP, MatrixProductBP.Models

include((@__DIR__)*"/../../src/mpbp.jl")

include((@__DIR__)*"/../../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

J = 1.0
β = 1.0
h = 0.0
k = 3

(m_ss, r_ss) = equilibrium_observables(RandomRegular(k), J; β, h)

ds = 1:10

maxiter = 70
tol = 1e-7
maxiter_vumps = 10
tol_vumps = 1e-14

spin(x,args...) = 3-2x
bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)

m_bp, r_bp = map(eachindex(ds)) do a
    only(bp.μ).tensor = reshape([0.1 0.1; 0.1 10], 1, 1, 2, 2)
    svd_trunc = TruncVUMPS(ds[a]; maxiter=maxiter_vumps, tol=tol_vumps)
    cb = CB_BPVUMPS(bp; f=spin)
    iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
    iter == maxiter && @warn "BP didn't converge"
    m = only(only(means(spin, bp)))
    r = only(only(alternate_correlations(spin, bp)))
    m, r
end |> unzip

jldsave((@__DIR__)*"/../../data/glauber_ferro2.jld2"; J, h, β, ds, m_bp, r_bp, m_ss, r_ss)

@telegram "vumps glauber ferro finished"