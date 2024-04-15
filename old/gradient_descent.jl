using MPSExperiments
using Test

using TensorTrains, Random, Tullio, TensorCast, ProgressMeter

rng = MersenneTwister(0)
qs = (4)

L = 15
M = rand(rng, 15, 15, qs...)
p = UniformTensorTrain(M, L)
normalize!(p)
sz = 5
η = 1e-2
maxiter = 500

A = rand(sz, sz, qs...)
normalize!(A)
q = UniformTensorTrain(A, L)
g = copy(A)

for it in 1:maxiter
    gradientA!(g, p, q)
    A .-= η * norm(A) * reshape(g, size(A))
    normalize!(q)
    if mod(it, 10) == 0 
        n = norm2m(p, q)
        m = norm(marginals(p)[1] - marginals(q)[1])
        println("-- Iter $it --")
        println("∑ₓ(q-p)^2 = ", n)
        println("Error on marginals = ", m)
    end
end

