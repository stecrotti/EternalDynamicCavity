using InfiniteMatrixProductBP
using MatrixProductBP, MatrixProductBP.Models
using IndexedGraphs, Statistics
using ProgressMeter
import Measurements: value, ±
using Graphs
using JLD2
using Unzip

include((@__DIR__)*"/meanfield.jl")


T_mf = 10^4     # final time
ρ = 0.1   # prob. of recovery
γ = 0.5    # prob. of being zero patient
k = 3

λs = 0.03:0.005:0.09

N = 5*10^3
graph_seed = 0
g = IndexedGraph(random_regular_graph(N, k, seed=graph_seed))

p_dmp_disc, p_ibmf_disc, p_cme_disc = map(λs) do λ
    p_dmp_disc, = dmp_disc(T_mf, 1.0, g, λ, ρ, fill(γ, nv(g)))     
    p_ibmf_disc = ibmf_disc(T_mf, 1.0, g, λ, ρ, fill(γ, nv(g)))
    p_cme_disc, = cme_disc(T_mf, 1.0, g, λ, ρ, fill(γ, nv(g)))
    p_dmp_disc, p_ibmf_disc, p_cme_disc
end |> unzip

p_dmp_disc_avg = mean.(p_dmp_disc)
p_ibmf_disc_avg = mean.(p_ibmf_disc)
p_cme_disc_avg = mean.(p_cme_disc)

#### MONTECARLO
Ts = fill(4000, length(λs))
nsamples = 50
@assert length(Ts) == length(λs)

p_mc = map(eachindex(λs)) do a
    λ = λs[a]
    sis = SIS(g, λ, ρ, Ts[a]; γ=fill(γ, N))
    bp = mpbp(sis)
    sms = SoftMarginSampler(bp)
    sample!(sms, nsamples)
    m_mc = means((x, args...) -> x-1, sms)
    p_mc = mean_with_uncertainty(m_mc)
    println("λ=$λ done. $a of $(length(λs))")
    p_mc
end

p_mc_val = [value.(p) for p in p_mc]


#### BPVUMPS
maxiter = 70
tol = 1e-5
maxiter_vumps = 10
tol_vumps = 1e-14

ds = fill(14, length(λs))
ds[6] = 20
@assert length(ds) == length(λs)

function main_loop()
    A0 = reshape([10 10; 0.1 0.1], 1,1,2,2)
    map(eachindex(λs)) do a
        λ = λs[a]
        svd_trunc = TruncVUMPS(ds[a]; maxiter=maxiter_vumps, tol=tol_vumps)
        # do some iterations with autoinfection to direct BP away from the absorbing states "all susceptible"
        bp_pre = mpbp_stationary_infinite_graph(k, [SISFactor(λ, ρ; α=0.1)], 2)
        only(bp_pre.μ).tensor = A0
        iterate!(bp_pre; maxiter, svd_trunc, tol)
        bp = mpbp_stationary_infinite_graph(k, [SISFactor(λ, ρ; α=0.0)], 2)
        bp.μ[1] = bp_pre.μ[1]
        cb = CB_BPVUMPS(bp; f = (x, args...) -> x-1)
        iter, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
        A0 = deepcopy(only(bp.μ).tensor)
        iter, cb
    end |> unzip
end

iters, cbs = main_loop()

p_bp = [[only(only(m)) for m in cb.m][end] for cb in cbs]


#### SAVE
jldsave("../../data/sis_meanfield_avg3.jld2"; T_mf, Ts, ds, λs, ρ, γ, N, graph_seed, nsamples,
    p_dmp_disc_avg, p_ibmf_disc_avg, p_cme_disc_avg,
    p_mc_val, p_bp)