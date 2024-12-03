J = [0 1 0 0
     1 0 1 1
     0 1 0 0
     0 1 0 0] .|> float

N = size(J, 1)
h = ones(N)

β = 1.0
ising = Ising(J, h, β)
gl = Glauber(ising, 0)
bp = mpbp_stationary(gl)

spin(x, args...) = 3-2x
svd_trunc = TruncVUMPS(5)
iterate!(bp; tol=1e-14, maxiter=10, svd_trunc)
m_bp = [only(m) for m in means(spin, bp)]

bp_slow = MPBP(bp.g, [GenericFactor.(w) for w in bp.w], bp.ϕ, bp.ψ, 
    deepcopy(collect(bp.μ)), deepcopy(bp.b), collect(bp.f))
iterate!(bp_slow; tol=1e-14, maxiter=10, svd_trunc)
m_bp_slow = [only(m) for m in means(spin, bp_slow)]
@test m_bp_slow ≈ m_bp