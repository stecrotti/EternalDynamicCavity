using JLD2, UnPack
using MatrixProductBP, MatrixProductBP.Models

data = load((@__DIR__)*"/../../data/sis_meanfield_avg3.jld2")
@unpack p_mc_val, λs, ρ = data
p_mc = last.(p_mc_val)

k = 3
ds = 1:14
tol_prob = 1e-3
αs = 10.0 .^ (-1:-2:-5)

maxiter = 100
tol = 1e-8
maxiter_vumps = 10
tol_vumps = 1e-14

function run_bp!(ps, errs, λ, p_true, ds, αs;
        A0 = reshape([0.1 0.1; 0.1 10], 1,1,2,2))
    bp = mpbp_stationary_infinite_graph(k, [SISFactor(λ, ρ)], 2)
    p = NaN
    for a in eachindex(ds)
        d = ds[a]
        svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
        for b in eachindex(αs)
            α = αs[b]
            bp = mpbp_stationary_infinite_graph(k, [SISFactor(λ, ρ; α)], 2)
            bp.μ[1].tensor = A0
            cb = CB_BPVUMPS(bp; f = (x, args...) -> x-1)
            iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
            iter ≥ maxiter && @warn "BP didn't converge"
            p = only(only(cb.m[end]))
            A0 = bp.μ[1].tensor
        end
        ps[a] = p
        er = abs(p - p_true)
        errs[a] = er
        println("\n\n d=$d, p=$p, err=$(er)\n\n")
        if er < tol_prob
            return d
        end
        p = NaN
    end
    return typemax(Int)
end

ps = [fill(NaN, length(ds)) for _ in λs]
errs = [fill(NaN, length(ds)) for _ in λs]

A0 = reshape([0.1 0.1; 0.1 10], 1,1,2,2)

for i in reverse(eachindex(λs))
    λ = λs[i]
    p_true = p_mc[i]
    run_bp!(ps[i], errs[i], λ, p_true, ds, αs; A0)
    println("###\nFinished λ=$λ, $i of ", length(λs), "\n")
end


jldsave((@__DIR__)*"/../../data/sis_bonddims4.jld2"; λs, ρ, k, tol_prob, ps, errs, 
    dmax, ds, αs)