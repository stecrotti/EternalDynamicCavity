using MPSExperiments
import TensorTrains: InfiniteUniformTensorTrain, marginals
using Random, LinearAlgebra

rng = MersenneTwister(0)
D = 10
d = 9
M = rand(rng, D, D, 2, 3)
p = InfiniteUniformTensorTrain(M)
q = truncate_eachtensor(p, d)
@show dot(p, q)

d1 = only(marginals(q)) - only(marginals(p)) |> real |> vec

q, ε, ovl = truncate_variational(p, d; maxiter=20)
@show dot(p, q)

d2 = only(marginals(q)) - only(marginals(p)) |> real |> vec

[d1 d2]
