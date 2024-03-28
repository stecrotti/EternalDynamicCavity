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


function F(λ, ρ)
    SUSCEPTIBLE = 1 
    INFECTIOUS = 2
    function f(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, xᵢᵗ::Integer)
        @assert xᵢᵗ⁺¹ ∈ 1:2
        @assert all(x ∈ 1:2 for x in xₙᵢᵗ)

        if xᵢᵗ == INFECTIOUS
            if xᵢᵗ⁺¹ == SUSCEPTIBLE
                return ρ
            else
                return 1 - ρ 
            end
        else
            p = (1-λ)^sum(xⱼᵗ == INFECTIOUS for xⱼᵗ in xₙᵢᵗ; init=0.0)
            if xᵢᵗ⁺¹ == SUSCEPTIBLE
                return p
            elseif xᵢᵗ⁺¹ == INFECTIOUS
                return 1 - p
            end
        end
    end   
end   


λ = 0.2
ρ = 0.2
f = F(λ, ρ)

# ds = [25, 30]
# ds = [5, 10, 15, 20, 25]
ds = [3, 4]
# ds = 5:3:20

A0 = reshape([0.210472  0.210472;  0.289528  0.289528], 1,1,2,2)

ε, err, ovl, bel = map(eachindex(ds)) do a
    d = ds[a]
    A, maxiter, εs, errs, ovls, beliefs = iterate_bp_vumps(f, d; A0, tol=1e-6, maxiter=60)
    @telegram "vumps sis $a/$(length(ds)) finished"
    εs, errs, ovls, beliefs
end |> unzip

using Plots

pls = map(zip(ε, err, ovl, ds, bel)) do (εs, errs, ovls, d, beliefs)
    replace!(ovls, 1.0 => NaN)
    p1 = plot(εs, xlabel="iter", yaxis=:log10, ylabel="converg error", label="")
    p2 = plot(errs, xlabel="iter", yaxis=:log10, ylabel="trunc err on marginals", label="", title="d=$d")
    p3 = plot(abs.(1 .- ovls), xlabel="iter", yaxis=:log10, ylabel="|1-trunc ovl|", label="")
    p4 = plot([real(b[2]) for b in beliefs], ylabel="Re[single-var prob(Infectious)]", ylims=(0,1), label="")
    hline!(p4, [0.5798385304677115] , label="steady-state", ls=:dash)
    plot(p1, p2, p3, p4, layout=(1,4), size=(1000,250), margin=5Plots.mm, labelfontsize=9)
end
pl = plot(pls..., layout=(length(ds),1), size=(1000, 250*length(ds)), margin=15Plots.mm)
savefig(pl, (@__DIR__)*"/truncations_vumps_sis3.pdf")