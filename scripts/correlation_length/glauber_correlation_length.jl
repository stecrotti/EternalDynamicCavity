using JLD2
using MatrixProductBP, MatrixProductBP.Models
using TensorTrains: one_normalization
using KrylovKit

using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

k = 3
β = 1.0
h = 0.0

Js = 0.1:0.05:1

function run_bp(J, d; A0=reshape([0.1 0.1; 0.1 10], 1,1,2,2))
    bp = mpbp_stationary_infinite_graph(k, [HomogeneousGlauberFactor(J, h, β)], 2)
    only(bp.μ).tensor = A0
    cb = CB_BPVUMPS(bp, f=spin)
    svd_trunc = TruncVUMPS(d; maxiter=maxiter_vumps, tol=tol_vumps)
    iters, cb = iterate!(bp; maxiter, svd_trunc, tol, cb)
    iters == maxiter && @warn "BP did not converge"
    return bp
end
function compute_corrlength(c, step, maxdist)
    c_bp = c[1,2:end][step:step:end]
    logc = log.([c > 0 ? c : NaN for c in c_bp])
    # logc = log.(abs.(c_bp))
    t = (1:maxdist)[step:step:end]
    sl =  t \ logc
    return -1/sl
end

function compute_corrlength_eigen(bp)
    ZA = one_normalization(only(bp.b))
    lambdas, = eigsolve(ZA, 2)
    @show abs(lambdas[2] / lambdas[1])
    return - 1 / log(abs(lambdas[2] / lambdas[1]))
end

function compute_magnetization(bp)
    m_bp = only(only(means(spin, bp)))
    return m_bp
end
function exact_magnetization(J)
    m, = equilibrium_observables(RandomRegular(k), J; h, β, tol=1e-8, maxiter=5*10^4)
    return m
end


ds = 3:17
maxiter = 20
tol = 1e-4
maxiter_vumps = 10
tol_vumps = 1e-14
spin(x, args...) = 3-2x

maxdist = 30
only_even = true
step = only_even + 1
t = (1:maxdist)[step:step:end]

bps = map(eachindex(Js)) do j
    J = Js[j]
    A0 = reshape([10 0.1; 0.1 0.1], 1,1,2,2)
    b = map(ds) do d
        bp = run_bp(J, d; A0)
        A0 = only(bp.μ).tensor
        bp
    end
    println("\n\tFinished $j of $(length(Js)): J=$J")
    b
end

corrs_ = reduce(hcat, [only.(autocovariances.(spin, b; maxdist)) for b in bps])
corrs = [corrs_[j,:] for j in eachindex(ds)]

ls = [compute_corrlength.(c, step, maxdist) for c in corrs]
ls_eigen_ = reduce(hcat, [compute_corrlength_eigen.(b) for b in bps])
ls_eigen = [ls_eigen_[j,:] for j in eachindex(ds)]
ms_ = reduce(hcat, [compute_magnetization.(b) for b in bps])
ms = [ms_[j,:] for j in eachindex(ds)]

ms_exact = exact_magnetization.(Js)

jldsave((@__DIR__)*"/../../data/glauber_correlation_length3.jld2"; Js, h, k, corrs, ds, ls, 
    ls_eigen, ms, ms_exact)