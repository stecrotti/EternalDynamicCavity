rng = MersenneTwister(0)
L = 12
A = rand(rng, 2,2,3,4)
p = UniformTensorTrain(A, L)
q = deepcopy(p)
orthogonalize_left!(q)
@test marginals(q) ≈ marginals(p)

p = InfiniteUniformTensorTrain(A)
q = deepcopy(p)
orthogonalize_left!(q)
@test marginals(q) ≈ marginals(p)
orthogonalize_right!(q)
@test marginals(q) ≈ marginals(p)

r = truncate_eachtensor(p, size(A, 1))
@test marginals(r) ≈ marginals(p)