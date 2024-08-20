using JLD2
using MatrixProductBP, MatrixProductBP.Models
using Plots, ColorSchemes
using LaTeXStrings
using Graphs, IndexedGraphs
using Measurements: value

include("../../../telegram/notifications.jl")

k = 3
J = 1.0
β = 1.0
h = 0.0

bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)

spin(x, i) = 3-2x
spin(x) = spin(x, 0)

T = 51
m⁰ = 0.5
nsamples = 10^4

function run_mc(N)
    g = random_regular_graph(N, k)
    ising = Ising(IndexedGraph(g); J=fill(J, ne(g)), h=fill(h, N), β)
    ϕᵢ = [ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T]
    bp_mc = mpbp(Glauber(ising, T); ϕ = fill(ϕᵢ, N))
    sms = SoftMarginSampler(bp_mc)

    sample!(sms, nsamples)

    X = autocovariances(spin, sms)
    c_mc = mean_with_uncertainty(X)

    return c_mc
end

Ns = [100, 200, 500, 1000, 10000]

c = run_mc.(Ns)

maxdist = 20
only_even = true
stp = only_even + 1
t = T - maxdist + 1
@assert t > 0

pl = plot()
cg = cgrad(:matter, length(Ns)+1, categorical=true)
ylims = (Inf, -Inf)
for a in eachindex(Ns)
    N = Ns[a]
    c_mc = c[a]
    c_mc_plot = map(x -> x<0 ? eltype(c_mc)(NaN) : x, c_mc[t,t+stp:stp:end])
    ylims_new = extrema(value.(c_mc_plot)) .* (2.0 .^ (-1,1))
    ylims = (min(ylims[1], ylims_new[1]), max(ylims[2], ylims_new[2]))
    scatter!(pl, 1:stp:maxdist, value.(c_mc_plot), c=cg[a+1],
        label="N=$N", m=:diamond, yaxis=:log10, legend=:bottomleft,
        ylims = ylims)
end  
savefig(pl, "glauber_autocovariance_ferro_montecarlo.pdf")

@telegram "Glauber autocovariance ferro montecarlo - script"

jldsave((@__DIR__)*"/../data/glauber_autocovariance_ferro_montecarlo.jld2"; J, h, β, k, Ns,
    c, maxdist, T, nsamples);