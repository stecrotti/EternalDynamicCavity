import Pkg; Pkg.activate((@__DIR__)*"../../..")
using JLD2
using MatrixProductBP, MatrixProductBP.Models
# BLAS.set_num_threads(4)

include("../../src/mpbp.jl")

include((@__DIR__)*"/../../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

k = 3

ρ = 0.1
Δts = 10.0 .^ (0:-0.25:-1.5)
# λs = [0.1, 0.2, 0.4]
# αs = [10.0 .^ (-1:-2:-5), 10.0 .^ (-1:-1:-3), 10.0 .^ (0:-2:-4)]
ds = [6, 8, 10]
ds = [2, 4, 6]
αs = fill(10.0 .^ (-1:-2:-5), length(ds))
λ = 0.4
# @assert length(ds) == length(λs) == length(αs)

maxiter = 100
tol = 1e-12
maxiter_vumps = 20
tol_vumps = 1e-14

p_bp = [zeros(length(Δts)) for _ in ds]

for (i, d) in enumerate(ds)
    println("### d=$d, bond dim $i of $(length(ds))")
    svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
    for (a, Δt) in enumerate(Δts)
        A0 = reshape([0.1 0.1; 0.1 10], 1,1,2,2)
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

jldsave((@__DIR__)*"/../../data/sis_continuous3.jld2"; k, λ, ρ, Δts, αs, ds, p_bp,
                                                       maxiter, tol, maxiter_vumps, tol_vumps)

# jldsave((@__DIR__)*"/../../data/sis_continuous.jld2"; k, λs, ρ, Δts, αs, ds, p_bp,
#                                                        maxiter, tol, maxiter_vumps, tol_vumps)

@telegram "SIS continuous"

# cbss = map(eachindex(Δts)) do a
#     Δt = Δts[a]
#     A0 = reshape([0.1 0.1; 0.1 10], 1,1,2,2)
#     cbs = map(eachindex(αs)) do b
#         α = αs[b]
#         bp = mpbp_stationary_infinite_graph(k, [SISFactor(λ*Δt, ρ*Δt; α=α*Δt)], 2)
#         bp.μ[1].tensor = A0
#         cb = CB_BPVUMPS(bp; f = (x, args...) -> x-1)
#         iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
#         A0 = bp.μ[1].tensor
#         cb
#     end
#     s = "SIS continuous - Finished round $a of $(length(Δts)) Δt=$(Δt). m=$(only(only(cbs[end].m[end])))"
#     println(s)
#     @telegram "$s"
#     cbs
# end

# p_bp = [only(only(cbs[end].m[end])) for cbs in cbss]

# λ = 0.1
# maxiter = 70
# tol = 1e-6
# maxiter_vumps = 20
# tol_vumps = 1e-14
# svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)

# cbss = map(eachindex(Δts)) do a
#     Δt = Δts[a]
#     A0 = reshape([0.1 0.1; 0.1 10], 1,1,2,2)
#     cbs = map(eachindex(αs)) do b
#         α = αs[b]
#         bp = mpbp_stationary_infinite_graph(k, [SISFactor(λ*Δt, ρ*Δt; α=α*Δt)], 2)
#         bp.μ[1].tensor = A0
#         cb = CB_BPVUMPS(bp; f = (x, args...) -> x-1)
#         iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
#         A0 = bp.μ[1].tensor
#         cb
#     end
#     s = "SIS continuous - Finished round $a of $(length(Δts)) Δt=$(Δt). m=$(only(only(cbs[end].m[end])))"
#     println(s)
#     @telegram "$s"
#     cbs
# end

# p_bp = [only(only(cbs[end].m[end])) for cbs in cbss]

# function main_loop()
#     A0 = reshape([0.1 0.1; 0.1 10], 1,1,2,2)

#     map(eachindex(Δts)) do a
#         Δt = Δts[a]; α = αs[a]
#         # do some iterations with autoinfection to direct BP away from the absorbing states "all susceptible"
#         bp_pre = mpbp_stationary_infinite_graph(k, [SISFactor(λ*Δt, ρ*Δt; α=α*Δt)], 2)
#         only(bp_pre.μ).tensor = A0
#         iterate!(bp_pre; maxiter, svd_trunc, tol)
#         bp = mpbp_stationary_infinite_graph(k, [SISFactor(λ*Δt, ρ*Δt; α=1e-2*Δt)], 2)
#         bp.μ[1].tensor = bp_pre.μ[1].tensor
#         cb = CB_BPVUMPS(bp; f = (x, args...) -> x-1)
#         iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
#         if iter == maxiter
#             @warn "BP not converged for Δt=$(Δt)"
#             @telegram "BP not converged for Δt=$(Δt)"
#         end
#         println("Finished round $a of $(length(Δts)) Δt=$(Δt)\n\n")
#         @telegram "SIS continuous - Finished round $a of $(length(Δts)) Δt=$(Δt). m=$(cb.m[end])"
#         A0 = only(bp.μ).tensor
#         iter, cb
#     end |> unzip
# end

# iters, cbs = main_loop()

# p_bp = [only(only(cb.m[end])) for cb in cbs]

# jldsave((@__DIR__)*"/../../data/sis_continuous_new.jld2"; k, λ, ρ, Δts, αs, d, p_bp,
#                                                        maxiter, tol, maxiter_vumps, tol_vumps)

# @telegram "SIS continuous"