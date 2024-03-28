using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter

include((@__DIR__)*"/../telegram/notifications.jl")

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

J = 0.1
β = 1.0
h = 0.2
f = F(J, β, h)

# ds = [25, 30]
# ds = [5, 10, 15, 20, 25]
ds = [3]
# ds = 5:3:25

A0 = reshape([0.31033984998979236 0.31033984998979214; 0.18966015001020783 0.1896601500102077], 1,1,2,2)

ε, err, ovl, bel = map(eachindex(ds)) do a
    d = ds[a]
    A, maxiter, εs, errs, ovls, beliefs = iterate_bp_vumps(f, d; A0, tol=1e-6, maxiter=30)
    # @telegram "vumps glauber $a/$(length(ds)) finished"
    εs, errs, ovls, beliefs
end |> unzip

using Plots

pls = map(zip(ε, err, ovl, ds, bel)) do (εs, errs, ovls, d, beliefs)
    replace!(ovls, 1.0 => NaN)
    p1 = plot(εs, xlabel="iter", yaxis=:log10, ylabel="converg error", label="")
    p2 = plot(errs, xlabel="iter", yaxis=:log10, ylabel="trunc err on marginals", label="", title="d=$d")
    p3 = plot(abs.(1 .- ovls), xlabel="iter", yaxis=:log10, ylabel="|1-trunc ovl|", label="")
    p4 = plot([real(reduce(-, b)) for b in beliefs], ylabel="Re[single-var magnetiz]", ylims=(-2,2), label="")
    hline!(p4, 0.8596433979493756 .* [-1, 1] , label="", ls=:dash)
    plot(p1, p2, p3, p4, layout=(1,4), size=(1000,250), margin=5Plots.mm, labelfontsize=9)
end
pl = plot(pls..., layout=(length(ds),1), size=(1000, 250*length(ds)), margin=15Plots.mm)
savefig(pl, (@__DIR__)*"/truncations_vumps_glauber2.pdf")

# @telegram "vumps glauber finished"