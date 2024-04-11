using MPSExperiments
using TensorTrains, LinearAlgebra, Random
using TensorTrains.UniformTensorTrains
using Unzip, Statistics
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using ProgressMeter
using MatrixProductBP, MatrixProductBP.Models

include((@__DIR__)*"/../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)

Random.seed!(1)


function F(λ, ρ; γ=0)
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
            p = (1-γ)*(1-λ)^sum(xⱼᵗ == INFECTIOUS for xⱼᵗ in xₙᵢᵗ; init=0.0)
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
# ds = [3, 4, 5, 6]
# ds = 5:5:10
ds = [3, 5]

maxiter = 50
tol = 1e-14

A0 = reshape([0.210472  0.210472;  0.289528  0.289528], 1,1,2,2)

ε, err, ovl, bel, AA, A = map(eachindex(ds)) do a
    d = ds[a]
    A, _, εs, errs, ovls, beliefs, As = iterate_bp_vumps(f, d; A0, tol, maxiter)
    # @telegram "vumps sis $a/$(length(ds)) finished"
    εs, errs, ovls, beliefs, A, As
end |> unzip

using Plots

pls = map(zip(ε, err, ovl, ds, bel)) do (εs, errs, ovls, d, beliefs)
    p1 = plot(replace(εs, 0.0 => NaN), xlabel="iter", yaxis=:log10, ylabel="converg error", label="")
    p2 = plot(errs, xlabel="iter", yaxis=:log10, ylabel="trunc err on marginals", label="", title="d=$d")
    p3 = plot(abs.(1 .- replace(ovls, 1.0 => NaN)), xlabel="iter", yaxis=:log10, ylabel="|1-trunc ovl|", label="")
    p4 = plot([b[2] for b in beliefs], ylabel="p(xᵢ=INFECT)", ylims=(-1,3), label="")
    hline!(p4, [0.5798385304677115] , label="", ls=:dash)
    p5 = plot([minimum(b) for b in beliefs], ylabel="min marginal", label="")
    plot(p1, p2, p3, p4, p5, layout=(1,5), size=(1200,250), margin=5Plots.mm, labelfontsize=9)
end
pl = plot(pls..., layout=(length(ds),1), size=(1000, 250*length(ds)), margin=5Plots.mm,
    xticks = 0:(maxiter÷2):maxiter, xlabel="iter")
# savefig(pl, (@__DIR__)*"/truncations_vumps_sis1.pdf")
# @telegram "vumps sis convergence finished"