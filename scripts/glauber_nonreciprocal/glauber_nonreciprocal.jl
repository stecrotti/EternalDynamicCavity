using MPSExperiments
using TensorTrains.UniformTensorTrains
using MatrixProductBP, MatrixProductBP.Models
using IndexedGraphs, Graphs
using JLD2


include("random_bipartite_regular.jl")

nA = 400*3
nB = 300*3
kA = 3
kB = 4

β = 1.0
JA = 0.1
JB = 0.5
h = 0.2

n = nA + nB
A = random_bipartite_regular(nA, nB, kA, kB)
gg = BipartiteIndexedGraph(A)
g = IndexedBiDiGraph(adjacency_matrix(gg))


#### MONTECARLO
T = 50
w = map(vertices(g)) do i
    fill(HomogeneousGlauberFactor(i ≤ nA ? JA : JB, h, β), T+1)
end
m⁰ = 0.5
ϕ = fill([ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T], n)

bp = mpbp(g, w, fill(2, n), T; ϕ)
sms = SoftMarginSampler(bp)

sample!(sms, 5*10^2)

spin(x, i) = 3-2x
spin(x) = spin(x, 0)
# m_mc = [[mss.val for mss in ms] for ms in means(spin, sms)]
m_mc = means(spin, sms)


##### uMPBP
fA = HomogeneousGlauberFactor(JA, h)
fB = HomogeneousGlauberFactor(JB, h)

w = [[HomogeneousGlauberFactor(JA, h, β)], [HomogeneousGlauberFactor(JB, h, β)]]
bp = mpbp_stationary_infinite_bipartite_graph((kA,kB), w, (2,2))

maxiter = 20
tol = 1e-14
maxiter_vumps = 10
tol_vumps = 1e-15
d = 5
spin(x, i) = 3-2x
spin(x) = spin(x, 0)


svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
cb = CB_BPVUMPS(bp; f=spin)

iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)

m_bp = means(spin, bp)

m_mcA = mean_with_uncertainty(m_mc[1:nA])
m_mcB = mean_with_uncertainty(m_mc[nA+1:end])
m_mc_mean = mean_with_uncertainty(m_mc)


jldsave((@__DIR__)*"/../../data/glauber_nonreciprocal.jld2"; kA, kB, nA, nB, JA, JB, h, β,
    m_mcA, m_mcB, m_mc_mean, m_bp, T)