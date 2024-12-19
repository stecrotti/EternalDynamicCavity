using JLD2
using MatrixProductBP, MatrixProductBP.Models
using TensorTrains: one_normalization
using KrylovKit

include("../../../telegram/notifications.jl")

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

ρ = 0.1   # prob. of recovery
k = 3

λs = 0.01:0.005:0.09

function run_bp(λ, d; A0=reshape([0.1 0.1; 0.1 10], 1,1,2,2))
    bp = mpbp_stationary_infinite_graph(k, [SISFactor(λ, ρ)], 2)
    only(bp.μ).tensor = A0
    cb = CB_BPVUMPS(bp, f=spin)
    svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
    iters, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
    iters == maxiter && @warn "BP did not converge"
    return bp
end
si(x, args...) = x - 1

function compute_corrlength(c, step, maxdist)
    c_bp = c[1,2:end][step:step:end]
    # logc = log.([c > 0 ? c : NaN for c in c_bp])
    logc = log.(abs.(c_bp))
    t = (1:maxdist)[step:step:end]
    sl =  t \ logc
    return -1/sl
end

function compute_corrlength_eigen(bp)
    ZA = one_normalization(only(bp.b))
    lambda, = eigsolve(ZA, 2)
    return - 1 / log(abs(lambda[2] / lambda[1]))
end

function compute_magnetization(bp)
    m_bp = only(only(means(si, bp)))
    return m_bp
end


ds = 3:2:19
maxiter = 20
tol = 1e-4
maxiter_vumps = 10
tol_vumps = 1e-14
spin(x, args...) = 3-2x

maxdist = 30
only_even = true
step = only_even + 1
t = (1:maxdist)[step:step:end]

bps = map(eachindex(λs)) do j
    λ = λs[j]
    A0 = reshape([10 0.1; 0.1 0.1], 1,1,2,2)
    b = map(ds) do d
        bp = run_bp(λ, d; A0)
        A0 = only(bp.μ).tensor
        bp
    end
    println("\n\tFinished $j of $(length(λs)): λ=$λ")
    b
end

corrs_ = reduce(hcat, [only.(autocovariances.(si, b; maxdist)) for b in bps])
corrs = [corrs_[j,:] for j in eachindex(ds)]

ls = [compute_corrlength.(c, step, maxdist) for c in corrs]
ls_eigen_ = reduce(hcat, [compute_corrlength_eigen.(b) for b in bps])
ls_eigen = [ls_eigen_[j,:] for j in eachindex(ds)]
ms_ = reduce(hcat, [compute_magnetization.(b) for b in bps])
ms = [ms_[j,:] for j in eachindex(ds)]

jldsave((@__DIR__)*"/../../data/sis_correlation_length2.jld2"; λs, ρ, k, ds, corrs, ls, 
    ls_eigen, ms)

@telegram "SIS corr length"

using Plots, ColorSchemes, LaTeXStrings

step = 1
cg = cgrad(:matter, length(ds)+1, categorical=true)

pl = plot(; xlabel=L"λ", ylabel=L"\tau", title="Linear fit")
for a in eachindex(ds)[end:-step:1]
    plot!(pl, λs, ls[a], c=cg[a+1], label="d=$(ds[a])", m=:o)
end

pl_eigen = plot(; xlabel=L"λ", ylabel=L"\tau", title="Eigenvalues")
for a in eachindex(ds)[end:-step:1]
    plot!(pl_eigen, λs, ls_eigen[a], c=cg[a+1], label="d=$(ds[a])", m=:o)
end

pl2 = plot(pl, pl_eigen, layout=(2,1))
plot!(pl2, size=(600,600))
savefig(pl2, "sis_correlation_length.pdf")