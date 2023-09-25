using MPSExperiments
using Test
using TensorTrains, Random, Tullio, TensorCast, ProgressMeter

rng = MersenneTwister(0)
qs = (5)

M = rand(rng, 5, 5, qs...)
p = InfiniteUniformTensorTrain(M)
normalize!(p)

pfinite = UniformTensorTrain(M, 10)

rng = MersenneTwister(0)
sz = 3
truncate_utt(p, sz; damp=0.8, maxiter=1, showprogress=true)
truncate_utt(pfinite, sz; damp=0.8, maxiter=1, showprogress=true)
nothing

# sz = 14
# A = truncate_utt(p, sz; damp=0.8, maxiter=1, showprogress=true)
# q = InfiniteUniformTensorTrain(A)

# d = norm(marginals(q)[1] - marginals(p)[1])
# @show d

# A2 = truncate_utt_eigen(p, sz;  damp=0.8, maxiter=100, showprogress=true)
# q2 = InfiniteUniformTensorTrain(A2)
# d2 = norm(marginals(q2)[1] - marginals(p)[1])