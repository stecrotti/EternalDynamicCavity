using MPSExperiments, TensorTrains, Random
import TensorTrains: UniformTensorTrain, marginals, InfiniteUniformTensorTrain
import ForwardDiff
using DiffResults

function DifferenceNorm(q, p)
    sz = size(q.tensor)
    L = q.L

    function difference_norm(x)
        A = reshape(x, sz)
        q = UniformTensorTrain(A, L)
        return norm2m(q, p)
    end
end

function findmin_norm!(A, p; tol=1e-18, maxiter=100, η=1e-4, γ=1e-2,
        L = fill(NaN, maxiter+1))
    q = UniformTensorTrain(A, p.L)
    normalize!(q)
    for it in 1:maxiter
        foo = DifferenceNorm(q, p)
        res = DiffResults.GradientResult(A[:])
        ForwardDiff.gradient!(res, foo, A[:])
        L[it+1] = DiffResults.value(res)
        abs(L[it+1] - L[it]) < tol && return q, L
        dA = reshape(DiffResults.gradient(res), size(A))
        A .-= η * sign.(dA)
        η *= (1-γ)
        println("Iter $it of $maxiter: ℒ=$(L[it+1]), |∇L|=$(norm(dA)/length(dA))")
    end
    return q, L
end

rng = MersenneTwister(0)
D = 10
d = 7
M = rand(rng, D, D, 2, 3)
L = 5
p = UniformTensorTrain(M, L)
normalize!(p)

A = rand(rng, d, d, 2, 3)
# A0 = truncate_eachtensor(p, d).tensor
maxiter = 100
L = fill(NaN, maxiter+1)
q, L = findmin_norm!(A, p; η=1e-3, γ=1-(1e-10)^(1/maxiter), maxiter, L)

using Plots
display(plot(L, yaxis=:log10))

pp = twovar_marginals(p)
qq = twovar_marginals(q)
vcat([sum(abs, only(marginals(p)) - only(marginals(q)))],
    [sum(abs, pp[1,t] - qq[1,t]) for t in 2:p.L])