using MPSExperiments
using Test
using TensorTrains, Random, Tullio, TensorCast, ProgressMeter

rng = MersenneTwister(0)

M = rand(rng, 15, 15, qs...)
p = InfiniteUniformTensorTrain(M)

sz = 10
A = truncate_utt(p, sz; damp=0.8, maxiter=100, showprogress=true, rng)
q = InfiniteUniformTensorTrain(A)

d = norm(marginals(q)[1] - marginals(p)[1])
@show d

A2 = truncate_utt_eigen(p, sz;  damp=0.8, maxiter=100, showprogress=true, rng)
q2 = InfiniteUniformTensorTrain(A2)
d2 = norm(marginals(q2)[1] - marginals(p)[1])

# pL = UniformTensorTrain(M, L)
# AL = truncate_utt(p, sz; damp=0.5, maxiter=100, showprogress=true, rng)
# qL = UniformTensorTrain(A, L)
# d3 = norm(marginals(qL)[1] - marginals(pL)[1])


# d1, d2

# L = 3
# B = rand_periodic_tt(fill(3, L), 2)
# p = symmetrized_uniform_tensor_train(B)
# normalize!(p)
# M = p.tensor

# A = truncate_utt(p, size(M, 1); damp = 0.0, A0=M.+1e-10, 
#     maxiter=50)
# q = UniformTensorTrain(A, L)
# norm(marginals(q)[1] - marginals(p)[1])