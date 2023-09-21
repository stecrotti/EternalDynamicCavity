using TensorCast

rng = MersenneTwister(2)

L = 15
qs = (3, 2, 2)
M = rand(rng, 15, 15, qs...)
p = UniformTensorTrain(M, L)

sz = 5

# A0 = rand(rng, sz, sz, size(p.tensor)[3:end]...)
# A = truncate_utt(p, sz; damp=0.8)
# q = UniformTensorTrain(A, L)

# d1 = norm(marginals(q)[1] - marginals(p)[1])

A = truncate_utt_inf(p, sz; damp=0.8)
q = UniformTensorTrain(A, L)

d2 = norm(marginals(q)[1] - marginals(p)[1])

# d1, d2