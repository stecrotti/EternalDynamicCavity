using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter
using MatrixProductBP, MatrixProductBP.Models

include((@__DIR__)*"/../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)

function F(J, β, h)
    function f(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, xᵢᵗ::Integer)
        @assert xᵢᵗ⁺¹ ∈ 1:2
        @assert all(x ∈ 1:2 for x in xₙᵢᵗ)
        potts2spin(x) = 3-2x
        hⱼᵢ = β*J * sum(potts2spin, xₙᵢᵗ; init=0.0)
        E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + β*h)
        return 1 / (1 + exp(2E))
    end   
end   

J = 0.8
β = 1.0
h = 0.0
f = F(J, β, h)

(m_ss,) = equilibrium_observables(RandomRegular(3), J; β, h)

ds = 2:2:6

maxiter = 30
tol = 1e-15

# Random.seed!(3)
A0 = rand(1,1,2,2)
A0 = reshape([0.4 0.4; 0.2 0.2], 1, 1, 2, 2)

ε, err, ovl, bel, AA, A = map(eachindex(ds)) do a
    d = ds[a]
    A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(f, d; A0, tol, maxiter)
    εs, errs, ovls, beliefs, A, As
end |> unzip

using Plots

pls = map(zip(ε, err, ovl, ds, bel)) do (εs, errs, ovls, d, beliefs)
    p1 = plot(replace(εs, 0.0 => NaN), xlabel="iter", yaxis=:log10, ylabel="converg error", label="")
    p2 = plot(errs, xlabel="iter", yaxis=:log10, ylabel="trunc err on marginals", label="", title="d=$d")
    p3 = plot(abs.(1 .- replace(ovls, 1.0 => NaN)), xlabel="iter", yaxis=:log10, ylabel="|1-trunc ovl|", label="")
    p4 = plot([reduce(-, b) for b in beliefs], ylabel="single-var magnetiz", ylims=(-2,2), label="")
    hline!(p4, [m_ss] , label="", ls=:dash)
    plot(p1, p2, p3, p4, layout=(1,4), size=(1200,250), margin=5Plots.mm, labelfontsize=9)
end
pl = plot(pls..., layout=(length(ds),1), size=(1000, 250*length(ds)), margin=5Plots.mm,
    xticks = 0:(maxiter÷2):maxiter, xlabel="iter")
savefig(pl, (@__DIR__)*"/vumps_glauber.pdf")

ps = [reduce(-,b[findlast(x->!all(isnan, x), b)]) for b in bel]
pl_ps = scatter(ds, ps, xlabel="bond dim", ylabel="magnetiz", label="",
    ms=2, c=:black)
hline!(pl_ps, [m_ss], ylims=(0.9,1.), label="true steady-state")
plot!(pl_ps, title="Glauber J=$J, h=$h, β=$β")
savefig(pl_ps, "vumps_glauber_bonddims.pdf")

jldsave((@__DIR__)*"/../data/vumps_glauber.jld2"; J, h, β, ds, A0, ε, err, ovl, bel, AA, A, maxiter, ps, m_ss)

@telegram "vumps glauber finished"