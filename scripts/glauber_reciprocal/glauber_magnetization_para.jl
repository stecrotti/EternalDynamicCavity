using MPSExperiments
using Unzip, Statistics
using ProgressMeter
using JLD2
using MatrixProductBP, MatrixProductBP.Models

J = 0.4
β = 1.0
h = 0.2
k = 3

(m_ss, r_ss) = equilibrium_observables(RandomRegular(k), J; β, h, tol=1e-12)

ds = 1:10

maxiter = 150
tol = 1e-10
maxiter_vumps = 30
tol_vumps = 1e-14

spin(x,args...) = 3-2x
bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)

m_bp, r_bp, cbs = map(eachindex(ds)) do a
    only(bp.μ).tensor = reshape([0.1 0.1; 0.1 10], 1, 1, 2, 2)
    svd_trunc = TruncVUMPS(ds[a]; maxiter=maxiter_vumps, tol=tol_vumps)
    cb = CB_BPVUMPS(bp; f=spin)
    iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
    iter == maxiter && @warn "BP didn't converge"
    m = only(only(means(spin, bp)))
    r = only(only(alternate_correlations(spin, bp)))
    m, r, cb
end |> unzip

jldsave((@__DIR__)*"/../../data/glauber_para3.jld2"; J, h, β, ds, m_bp, r_bp, m_ss, r_ss)