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