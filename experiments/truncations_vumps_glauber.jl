using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter
using JLD2
using MatrixProductBP, MatrixProductBP.Models

include("../src/mpbp.jl")

# include((@__DIR__)*"/../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

J = 0.9
β = 1.0
h = 0.0
k = 3
T = 0

maxiter = 50
tol = 1e-15

maxiter_vumps = 100
tol_vumps = 1e-14

d = 10
svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
maxiter = 20
tol = 0

spin(x,args...) = 3-2x
bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)
cb = CB_BP(bp; f=spin)
iters, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)

(m_ss, r_ss) = equilibrium_observables(RandomRegular(3), J; β, h)

using Plots
pl1 = plot([only(only(m)) for m in cb.m], xlabel="iter", ylabel="magnetiz")
hline!(pl1, [m_ss, -m_ss], ls=:dash, ylims=(-1,1))

pl2 = hline([autocorrelations(spin, bp)], xlabel="iter", ylabel="2-time autocorr")
hline!(pl2, [r_ss], ls=:dash, ylims=(-1,1))

plot(pl1, pl2)

# ds = 1:2:13

# maxiter = 30
# tol = 1e-15
# maxiter_vumps = 10

# # Random.seed!(3)
# A0 = rand(1,1,2,2)
# A0 = reshape([0.4 0.4; 0.2 0.2], 1, 1, 2, 2)
# A_current = copy(A0)

# ε, err, ovl, bel, AA, A = map(eachindex(ds)) do a
#     d = ds[a]
#     global A_current, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(f, d; A0=A_current, tol, maxiter, maxiter_vumps)
#     εs, errs, ovls, beliefs, copy(A_current), As
# end |> unzip

# using Plots

# pls = map(zip(ε, err, ovl, ds, bel)) do (εs, errs, ovls, d, beliefs)
#     p1 = plot(replace(εs, 0.0 => NaN), xlabel="iter", yaxis=:log10, ylabel="converg error", label="")
#     p2 = plot(errs, xlabel="iter", yaxis=:log10, ylabel="trunc err on marginals", label="", title="d=$d")
#     p3 = plot(abs.(1 .- replace(ovls, 1.0 => NaN)), xlabel="iter", yaxis=:log10, ylabel="|1-trunc ovl|", label="")
#     p4 = plot([reduce(-, b) for b in beliefs], ylabel="single-var magnetiz", ylims=(-1.1,1.1), label="")
#     hline!(p4, [m_ss] , label="", ls=:dash)
#     plot(p1, p2, p3, p4, layout=(1,4), size=(1200,250), margin=5Plots.mm, labelfontsize=9)
# end
# pl = plot(pls..., layout=(length(ds),1), size=(1000, 250*length(ds)), margin=5Plots.mm,
#     xticks = 0:(maxiter÷2):maxiter, xlabel="iter")
# savefig(pl, (@__DIR__)*"/vumps_glauber3.pdf")

# ps = [reduce(-,b[findlast(x->!all(isnan, x), b)]) for b in bel]
# pl_ps = scatter(ds, ps, xlabel="bond dim", ylabel="magnetiz", label="",
#     ms=3, c=:black, xticks=ds)
# hline!(pl_ps, [m_ss], label="true steady-state")
# plot!(pl_ps, title="Glauber J=$J, h=$h, β=$β")
# savefig(pl_ps, "vumps_glauber_bonddims3.pdf")

# jldsave((@__DIR__)*"/../data/vumps_glauber.jld2"; J, h, β, ds, A0, ε, err, ovl, bel, AA, A, maxiter, ps, m_ss)

# @telegram "vumps glauber finished"