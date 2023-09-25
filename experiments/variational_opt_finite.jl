using MPSExperiments
using Test

using TensorTrains, Random, Tullio, TensorCast, ProgressMeter

rng = MersenneTwister(0)
qs = (4)

L = 50
M = rand(rng, 15, 15, qs...)
p = UniformTensorTrain(M, L)
normalize!(p)
sz = 14
A = truncate_utt(p, sz;  damp=0.8)
q = UniformTensorTrain(A, L)
d1 = norm(marginals(q)[1] - marginals(p)[1])