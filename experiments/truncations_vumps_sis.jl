using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter
using JLD2
using MatrixProductBP, MatrixProductBP.Models

using MPSKit, KrylovKit

include((@__DIR__)*"/../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)

λ = 0.07
ρ = 0.1
p0 = 0.9

ds = 6:2:16
# ds = [6]

maxiter = 50
tol = 1e-15
maxiter_vumps = 100
maxiter_ortho = 100; tol_ortho = 1e-8
maxiter_fixedpoint = 100; tol_fixedpoint = 1e-8
tol_vumps = 1e-12

Random.seed!(3)
A0 = rand(1,1,2,2)
A0 = reshape([10 10; 0.2 0.2], 1,1,2,2)

d = 6

stats = @timed begin
A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(F(λ, ρ; γ=5e-2), d; A0, tol, maxiter,
    maxiter_vumps=10, maxiter_ortho=10, maxiter_fixedpoint=10, tol_vumps, tol_ortho, 
    tol_fixedpoint)
#A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(F(λ, ρ; γ=5e-2), d; A0=A, tol, maxiter,
#    maxiter_vumps, maxiter_ortho, maxiter_fixedpoint, tol_vumps, tol_ortho, 
#    tol_fixedpoint)
Base.GC.gc()
state = VUMPSState(size(A0,1), d, 4)
A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(F(λ, ρ; γ=0.0), d; A0=A, tol=1e-7, maxiter,
    maxiter_vumps, maxiter_ortho, maxiter_fixedpoint, tol_vumps=1e-7, tol_ortho=1e-7,
    tol_fixedpoint=1e-7, state)

A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(F(λ, ρ; γ=0.0), d; A0=A, tol, maxiter,
    maxiter_vumps, maxiter_ortho, maxiter_fixedpoint, tol_vumps, tol_ortho,
    tol_fixedpoint, state)

end
@show stats.time

stats = @timed begin
    A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps_mpskit(F(λ, ρ; γ=5e-2), d; A0, tol, 
        maxiter, maxiter_vumps)
    Base.GC.gc()
    A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps_mpskit(F(λ, ρ; γ=0.0), d; A0=A, tol, 
        maxiter, maxiter_vumps)
end
@show stats.time






# stats = @timed begin
ε, err, ovl, bel, AA, A = map(eachindex(ds)) do a
    d = ds[a]
    f = DampedFactor(SISFactor(λ, ρ; α=1e-2), p0)
    A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(f, d; A0, tol, 
        maxiter, maxiter_vumps#=, alg_gauge, alg_eigsolve, alg_environments=#)
    Base.GC.gc()
    f = DampedFactor(SISFactor(λ, ρ; α=0.0), p0)
    A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(f, d; A0=A, tol, maxiter,
        maxiter_vumps#=, alg_gauge, alg_eigsolve, alg_environments=#)
    println("\n####\nBond dim d=$d, $a/$(length(ds))\n#####\n")
    εs, errs, ovls, beliefs, A, As
end |> unzip
# end
# @show stats.time

# using Plots

p_ss_montecarlo = 0.352887587128715
p_ss_mpbp = 0.35301856060787556
p_ss_gillespie = 0.33834699603960394
p_ss_gillespie = 0.33788149148514857

x = λ / ρ
k = 3
p_ss_cme = (x*(k-1)-1) / (x*(k-1)-1 + (k-1)/k)

pls = map(zip(ε, err, ovl, ds, bel)) do (εs, errs, ovls, d, beliefs)
    p1 = plot(replace(εs, 0.0 => NaN), xlabel="iter", yaxis=:log10, ylabel="converg error", label="")
    p2 = plot(errs, xlabel="iter", yaxis=:log10, ylabel="trunc err on marginals", label="", title="d=$d")
    p3 = plot(abs.(1 .- replace(ovls, 1.0 => NaN)), xlabel="iter", yaxis=:log10, ylabel="|1-trunc ovl|", label="")
    p4 = plot([b[2] for b in beliefs], ylabel="p(xᵢ=INFECT)", ylims=(0,1), label="")
    # hline!(p4, [0.5798385304677115] , label="", ls=:dash)
    # hline!(p4, [p_ss_montecarlo] , label="montecarlo", ls=:dash)
    # hline!(p4, [p_ss_mpbp] , label="MPBP", ls=:dash)
    hline!(p4, [p_ss_cme] , label="CME", ls=:dash)
    hline!(p4, [p_ss_gillespie] , label="gillespie", ls=:dash)
    # p5 = plot([minimum(b) for b in beliefs], ylabel="min marginal", label="")
    plot(p1, p2, p3, p4, layout=(1,4), size=(1200,250), margin=5Plots.mm, labelfontsize=9)
end
pl = plot(pls..., layout=(length(ds),1), size=(1000, 250*length(ds)), margin=5Plots.mm,
    xticks = 0:(maxiter÷2):maxiter, xlabel="iter")
savefig(pl, (@__DIR__)*"/vumps_sis.pdf")


ps = [b[findlast(x->!all(isnan, x), b)][2] for b in bel]
pl_ps = scatter(ds, ps, xlabel="bond dim", ylabel="p(xᵢ=INFECT)", label="",
    ms=2, c=:black)
hline!(pl_ps, [p_ss_cme] , label="CME", ylims=(0,1))
hline!(pl_ps, [p_ss_gillespie] , label="gillespie",)
plot!(pl_ps, title="λ=$λ, ρ=$ρ")
savefig(pl_ps, "vumps_sis_bonddims.pdf")

jldsave((@__DIR__)*"/../data/vumps_sis.jld2"; λ, ρ, ds, A0, ε, err, ovl, bel, AA, A, maxiter, ps, p_ss_montecarlo, p_ss_mpbp)

# # @telegram "vumps sis finished"