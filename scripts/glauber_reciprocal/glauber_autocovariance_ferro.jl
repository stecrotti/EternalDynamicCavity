using MPSExperiments
using TensorTrains.UniformTensorTrains
using JLD2
using MatrixProductBP, MatrixProductBP.Models
using Plots
using LaTeXStrings
using Graphs, IndexedGraphs

include("../../src/mpbp.jl")

include("../../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)


k = 3
J = 1.0
β = 1.0
h = 0.0

bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)

maxiter = 20
tol = 1e-5
maxiter_vumps = 10
tol_vumps = 1e-14
d = 14
spin(x, i) = 3-2x
spin(x) = spin(x, 0)

only(bp.μ).tensor = reshape([10 10; 0.1 0.1], 1, 1, 2, 2)
svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
cb = CB_BPVUMPS(bp)

iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)

maxdist = 20
c_bp = only(autocovariances(spin, bp; maxdist))[1,2:end]
only_even = true
step = only_even + 1
c_bp_plot = c_bp[step:step:end]
pl_c = plot(1:step:maxdist, c_bp_plot, label="MPBP+VUMPS")
plot!(pl_c; xlabel=L"\Delta t", size=(400,300),
    ylabel=L"\langle \sigma_i^t\sigma_i^{t+\Delta t}\rangle - \langle \sigma_i^t\rangle\langle\sigma_i^{t+\Delta t}\rangle")
plot!(pl_c; title="Time autocovariance. J=$J, h=$h, k=$k", titlefontsize=10)

#### MONTECARLO
T = 51
N = 5*10^3
m⁰ = 0.5
g = random_regular_graph(N, k)
ising = Ising(IndexedGraph(g); J=fill(J, ne(g)), h=fill(h, N), β)
ϕᵢ = [ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T]
bp_mc = mpbp(Glauber(ising, T); ϕ = fill(ϕᵢ, N))
sms = SoftMarginSampler(bp_mc)

sample!(sms, 10^4)

X = autocovariances(spin, sms)
c_mc = mean_with_uncertainty(X)

@telegram "Glauber autocovariance ferro - script"

jldsave((@__DIR__)*"/../data/glauber_autocovariance_ferro.jld2"; J, h, β, k, d, c_bp, c_mc, maxdist, T);