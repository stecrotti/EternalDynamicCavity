# using TensorCast

rng = MersenneTwister(0)
qs = (4)

L = 5
M = rand(rng, 15, 15, qs...)
p = UniformTensorTrain(M, L)
sz = 8
A = truncate_utt(p, sz; rng, damp=0.5)
q = UniformTensorTrain(A, L)
d1 = norm(marginals(q)[1] - marginals(p)[1])




# L = 100
# M = rand(rng, 15, 15, qs...)
# p = InfiniteUniformTensorTrain(M)

# sz = 5
# A = truncate_utt(p, sz; damp=0.5, maxiter=100, showprogress=true, rng)
# q = InfiniteUniformTensorTrain(A)

# d2 = norm(marginals(q)[1] - marginals(p)[1])
# @show d2

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