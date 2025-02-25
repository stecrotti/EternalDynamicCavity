using Unzip, Statistics
using ProgressMeter
using JLD2
using MatrixProductBP, MatrixProductBP.Models


J = 1.0
β = 1.0
h = 0.0
k = 3

(m_ss, r_ss) = equilibrium_observables(RandomRegular(k), J; β, h)

ds = 1:15

maxiter = 150
tol = 1e-12
maxiter_vumps = 50
tol_vumps = 1e-12

spin(x,args...) = 3-2x
bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)

function run_bp()
    A0 = reshape([0.1 0.1; 0.1 10], 1, 1, 2, 2)

    m_bp, r_bp, cbs = map(eachindex(ds)) do a
        only(bp.μ).tensor = A0
        svd_trunc = TruncVUMPS(ds[a]; maxiter=maxiter_vumps, tol=tol_vumps)
        cb = CB_BPVUMPS(bp; f=spin)
        iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
        iter == maxiter && @warn "BP didn't converge"
        m = only(only(means(spin, bp)))
        r = only(only(alternate_correlations(spin, bp)))
        A0 = only(bp.μ).tensor
        println("\nFinished d=$(ds[a]), $a of $(length(ds))\n")
        m, r, cb
    end |> unzip
end

m_bp, r_bp, cbs = run_bp()

jldsave((@__DIR__)*"/../../data/glauber_ferro5.jld2"; J, h, β, ds, m_bp, r_bp, m_ss, r_ss)