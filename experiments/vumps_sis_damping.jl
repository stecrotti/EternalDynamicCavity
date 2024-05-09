using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using MatrixProductBP, MatrixProductBP.Models
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter
using JLD2
using Plots

using MPSKit, KrylovKit

include((@__DIR__)*"/../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)

λ = 0.07
ρ = 0.1
d = 12
Δts = 10.0 .^ (0:-0.5:-3)
# damps = 1 .- 10.0 .^ (-1:-1:-3)
# damps = [0.1, 0.5, 0.9]
# damps = [0.9]

maxiter = 50
tol = 1e-12
maxiter_vumps = 100
Random.seed!(3)
A0 = rand(1,1,2,2)
A0 = reshape([0.5 0.5; 0.2 0.2], 1,1,2,2)

# stats = @timed begin
ε, err, ovl, bel, AA, A = map(eachindex(Δts)) do a
    Δt = Δts[a]
    f = SISFactor(λ*Δt, ρ*Δt; α=1e-2)
    A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps_mpskit(f, d; A0, tol, 
        maxiter, maxiter_vumps)
    Base.GC.gc()
    f = SISFactor(λ*Δt, ρ*Δt; α=0.0)
    A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps_mpskit(f, d; A0=A, tol, maxiter,
        maxiter_vumps)
    println("\n####\nΔt=$Δt, $a/$(length(Δts))\n#####\n")
    εs, errs, ovls, beliefs, A, As
end |> unzip
# end
# @show stats.time

# using Plots

# p_ss_montecarlo = 0.352887587128715
# p_ss_mpbp = 0.35301856060787556
p_gillespie = 0.33795382376237626


x = λ / ρ
k = 3
p_ss_cme = (x*(k-1)-1) / (x*(k-1)-1 + (k-1)/k) 

pls = map(zip(ε, err, ovl, Δts, bel)) do (εs, errs, ovls, Δt, beliefs)
    p1 = plot(replace(εs, 0.0 => NaN), xlabel="iter", yaxis=:log10, ylabel="converg error", label="")
    p2 = plot(errs, xlabel="iter", yaxis=:log10, ylabel="trunc err on marginals", label="", title="Δt=$Δt")
    p3 = plot(abs.(1 .- replace(ovls, 1.0 => NaN)), xlabel="iter", yaxis=:log10, ylabel="|1-trunc ovl|", label="")
    p4 = plot([b[2] for b in beliefs], ylabel="p(xᵢ=INFECT)", ylims=(0,1), label="")
    # hline!(p4, [0.5798385304677115] , label="", ls=:dash)
    # hline!(p4, [p_ss_montecarlo] , label="montecarlo", ls=:dash)
    # hline!(p4, [p_ss_mpbp] , label="MPBP", ls=:dash)
    hline!(p4, [p_ss_cme], label="CME", ls=:dash)
    hline!(p4, [p_gillespie], label="gillespie", ls=:dash)
    # p5 = plot([minimum(b) for b in beliefs], ylabel="min marginal", label="")
    pp = plot(p1, p2, p3, p4, layout=(1,4), size=(1200,250), margin=5Plots.mm, labelfontsize=9)
    # savefig(pp, (@__DIR__)*"/vumps_sis_damping2_$Δt.pdf")
    pp
end
pl = plot(pls..., layout=(length(Δts),1), size=(1000, 250*length(Δts)), margin=5Plots.mm,
    xticks = 0:(maxiter÷2):maxiter, xlabel="iter")
savefig(pl, (@__DIR__)*"/vumps_sis_damping2.pdf")


ps = [b[findlast(x->!all(isnan, x), b)][2] for b in bel]
pl_ps = scatter(Δts, ps, xlabel="Δt", ylabel="p(xᵢ=INFECT)", label="",
    ms=2, c=:black, ylims=(0,1))
hline!(pl_ps, [p_ss_cme], label="CME", ls=:dash)
hline!(pl_ps, [p_gillespie], label="gillespie", ls=:dash)
plot!(pl_ps, title="λ=$λ, ρ=$ρ, d=$d", xaxis=:log10)
savefig(pl_ps, "vumps_sis_damping_dampings2.pdf")

jldsave((@__DIR__)*"/../data/vumps_sis_damping.jld2"; λ, ρ, Δts, A0, ε, err, ovl, bel, AA, A, maxiter, ps)

@telegram "vumps sis damping finished"