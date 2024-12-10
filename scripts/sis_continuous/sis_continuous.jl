using JLD2
using MatrixProductBP, MatrixProductBP.Models
using LinearAlgebra
BLAS.set_num_threads(1)

k = 3

ρ = 0.1
Δts = 10.0 .^ (0:-0.1:-1.2)
λs = [0.075, 0.1, 0.2, 0.4]
αs = [10.0 .^ (-1:-2:-5), 10.0 .^ (-1:-2:-5), 10.0 .^ (-1:-1:-3), 10.0 .^ (0:-2:-4)]
ds = [6, 6, 6, 6]
@assert length(ds) == length(λs) == length(αs)

maxiter = 100
tol = 1e-12
maxiter_vumps = 20
tol_vumps = 1e-14

p_bp = [zeros(length(Δts)) for _ in λs]

for (i, λ) in enumerate(λs)
    println("### λ=$λ, $i of $(length(λs))")
    d = ds[i]
    svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
    A0 = reshape([0.1 0.1; 0.1 10], 1,1,2,2)

    for (a, Δt) in enumerate(Δts)
        p = 0.0
        for α in αs[i]
            bp = mpbp_stationary_infinite_graph(k, [SISFactor(λ*Δt, ρ*Δt; α=α*Δt)], 2)
            bp.μ[1].tensor = A0
            cb = CB_BPVUMPS(bp; f = (x, args...) -> x-1)
            iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
            A0 = bp.μ[1].tensor
            p = only(only(cb.m[end]))
        end
        p_bp[i][a] = p
        s = "SIS continuous - Finished round $a of $(length(Δts)) Δt=$(Δt). p=$p"
        println(s)
    end

end

jldsave((@__DIR__)*"/../../data/sis_continuous4.jld2"; k, λs, ρ, Δts, αs, ds, p_bp,
                                                       maxiter, tol, maxiter_vumps, tol_vumps)