using MPSExperiments
using Test
using TensorTrains, Random, Tullio, TensorCast, ProgressMeter

rng = MersenneTwister(0)
qs = (5)

M = rand(rng, 15, 15, qs...)
p = InfiniteUniformTensorTrain(M)
p.tensor ./= sqrt(abs(tr(infinite_transfer_operator(p))))

pfinite = UniformTensorTrain(M, 30)

rng = MersenneTwister(0)
sz = 6
A = truncate_utt(p, sz; damp=0.8, maxiter=500, showprogress=true, rng)
q = InfiniteUniformTensorTrain(A)
d1 = norm(marginals(q)[1] - marginals(p)[1])

A = truncate_utt_eigen(p, sz; damp=0.8, maxiter=500, showprogress=true, rng)
q = InfiniteUniformTensorTrain(A)
d2 = norm(marginals(q)[1] - marginals(p)[1])

d1, d2