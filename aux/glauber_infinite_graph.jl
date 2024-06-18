using MatrixProductBP, MatrixProductBP.Models
using Plots
using Random
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using JLD2
using LaTeXStrings

include("../../telegram/notifications.jl");

T = 150         # final time
k = 3          # degree
m⁰ = 0.3       # magnetization at time zero

β = 1.0
J = 0.6
h = 0.0;

wᵢ = fill(HomogeneousGlauberFactor(J, h, β), T+1)
ϕᵢ = [ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T]
bp = mpbp_infinite_graph(k, wᵢ, 2, ϕᵢ)
μ = only(bp.μ)
for t in eachindex(μ)
    rand!(MersenneTwister(t), μ[t])
end


cb = CB_BP(bp);

matrix_sizes = [10, 15]
maxiters = fill(50, length(matrix_sizes))
maxiters = [50, 9]
iters = zeros(Int, length(maxiters))
tol = 1e-5
for i in eachindex(maxiters)
    iters[i], _ = iterate!(bp; maxiter=maxiters[i], svd_trunc=TruncBond(matrix_sizes[i]), cb, tol)
end

iters_cum = cumsum(iters)
inds = 1:iters_cum[1]
pl_conv = plot(inds, cb.Δs[inds], label="$(matrix_sizes[1]) matrices")
for i in 2:length(iters)
    inds = iters_cum[i-1]:iters_cum[i]
   plot!(pl_conv, inds, cb.Δs[inds], label="$(matrix_sizes[i]) matrices")
end
plot(pl_conv, ylabel="convergence error", xlabel="iters", yaxis=:log10, size=(500,300),
    legend=:outertopright)

spin(x, i) = 3-2x
spin(x) = spin(x, 0)
m = only(means(spin, bp));

m_ss, = equilibrium_observables(RandomRegular(k), J; β, h)

blue = theme_palette(:auto)[1]
pl = plot(0:T, map(spin, only(cb.m[end])), m=:o, xlabel="time", ylabel="magnetization", label="MPBP",
    size=(500,300), xticks=0:20:T, ms=3, title="Glauber infinite $k-regular", titlefontsize=12,
    legend=:bottomright, msc=:auto, c=blue)
hline!(pl, [m_ss, -m_ss], ls=:dash, label="equilib")

@telegram "glauber infinite"

# using Graphs, IndexedGraphs, Statistics

# N = 10^3
# g = random_regular_graph(N, k)
# ising = Ising(IndexedGraph(g); J=fill(J, ne(g)), h=fill(h, N), β)
# bp_mc = mpbp(Glauber(ising, T); ϕ = fill(ϕᵢ, N))
# sms = SoftMarginSampler(bp_mc);