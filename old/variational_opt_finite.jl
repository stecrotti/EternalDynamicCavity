using MPSExperiments
using Test
using TensorTrains, Random, Tullio, TensorCast, ProgressMeter
using LinearAlgebra

rng = MersenneTwister(0)
qs = (8)

L = 5
d = 20
M = rand(rng, d, d, qs...)
p = UniformTensorTrain(M, L)
normalize!(p)
sz = 3
A = truncate_utt(p, sz; damp=0.8)
q = UniformTensorTrain(A, L)
d1 = norm(marginals(q)[1] - marginals(p)[1])