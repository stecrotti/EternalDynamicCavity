using MPSExperiments
using JLD2
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs

k = 3
J = 1.0
β = 1.0
h = 0.0

spin(x, args...) = 3-2x

T = 101
m⁰ = 0.5
nsamples = 10^4

function run_mc(N, maxdist)
    g = random_regular_graph(N, k)
    ising = Ising(IndexedGraph(g); J=fill(J, ne(g)), h=fill(h, N), β)
    ϕᵢ = [ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T]
    bp_mc = mpbp(Glauber(ising, T); ϕ = fill(ϕᵢ, N))
    sms = SoftMarginSampler(bp_mc)

    sample!(sms, nsamples)

    X = autocovariances(spin, sms; maxdist)
    c_mc = mean_with_uncertainty(X)

    return c_mc
end

Ns = [100, 200, 500, 1000, 10000]

maxdist = 40
c = run_mc.(Ns, maxdist)

jldsave((@__DIR__)*"/../../data/glauber_autocovariance_ferro_montecarlo2.jld2"; J, h, β, k, Ns,
    c, T, nsamples);