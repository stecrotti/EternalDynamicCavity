using JLD2
using MatrixProductBP, MatrixProductBP.Models

include("../../src/mpbp.jl")

# include("../../../telegram/notifications.jl")

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

maxdist = 40
c_bp = only(autocovariances(spin, bp; maxdist))[1,2:end]

# @telegram "Glauber autocovariance ferro - script"

jldsave((@__DIR__)*"/../../data/glauber_autocovariance_ferro.jld2"; J, h, β, k, d, c_bp, maxdist);