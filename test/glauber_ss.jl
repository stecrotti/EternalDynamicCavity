include("../src/mpbp.jl")

using Random
using MatrixProductBP, MatrixProductBP.Models
using MPSExperiments
using Test

rng = MersenneTwister(111)

J = [0 1 0 0
     1 0 1 1
     0 1 0 0
     0 1 0 0] .|> float

N = size(J, 1)
h = randn(rng, N)

β = 1.0
ising = Ising(J, h, β)
gl = Glauber(ising, 0)
bp = mpbp_stationary(gl)

spin(x, i) = 3-2x; spin(x) = spin(x, 0)
svd_trunc = TruncVUMPS(5)
iterate!(bp; tol=1e-14, maxiter=10, svd_trunc)
m_bp = [only(m) for m in means(spin, bp)]

m_exact = [0.39599264460505396, 0.0639884139080279, 0.6728494266992204, -0.029151110808061487]
@test m_bp ≈ m_exact

bp_slow = MPBP(bp.g, [GenericFactor.(w) for w in bp.w], bp.ϕ, bp.ψ, 
    deepcopy(collect(bp.μ)), deepcopy(bp.b), collect(bp.f))
iterate!(bp_slow; tol=1e-14, maxiter=10, svd_trunc)
m_bp_slow = [only(m) for m in means(spin, bp_slow)]
@test m_bp_slow ≈ m_exact