using MPSExperiments, TensorTrains, Random
import TensorTrains: UniformTensorTrain, marginals, InfiniteUniformTensorTrain
import ForwardDiff
using DiffResults

function DifferenceMarginals(q, p)
    sz = size(q.tensor)
    L = q.L
    pi = twovar_marginals(p)[1,2]
    # pi = first(marginals(p))

    function difference_marginals(x)
        A = reshape(x, sz)
        q = UniformTensorTrain(A, L)
        # qi = first(marginals(q))
        qi = twovar_marginals(q)[1,2]
        return sum(abs, qi - pi)
    end
end

function findmin_marginals!(A, p; tol=1e-18, maxiter=100, η=1e-4, γ=1e-2,
        L = fill(NaN, maxiter+1))
    q = UniformTensorTrain(A, p.L)
    normalize!(q)
    for it in 1:maxiter
        foo = DifferenceMarginals(q, p)
        res = DiffResults.GradientResult(A[:])
        ForwardDiff.gradient!(res, foo, A[:])
        L[it+1] = DiffResults.value(res)
        abs(L[it+1] - L[it]) < tol && return q, L
        dA = reshape(DiffResults.gradient(res), size(A))
        A .-= η * sign.(dA) .* abs.(A)
        η *= (1-γ)
        println("Iter $it of $maxiter: ℒ=$(L[it+1]), |∇L|=$(norm(dA)/length(dA))")
    end
    return q, L
end

rng = MersenneTwister(0)
D = 10
d = 6
M = rand(rng, D, D, 2, 3)
L = 5
p = UniformTensorTrain(M, L)
normalize!(p)

# A = truncate_eachtensor(p, d).tensor
A = rand(d, d, 2, 3)

q = UniformTensorTrain(A, p.L)
maxiter = 300
L = fill(NaN, maxiter+1)
normalize!(q)
nn = norm2m(p, q)

findmin_marginals!(A, p; η=1e-1, γ=1-(1e-5)^(1/maxiter), maxiter, L)

using UnicodePlots
display(lineplot(L, yscale=:log10))

normalize!(p)
normalize!(q)
@show nn, norm2m(p, q)
pp = twovar_marginals(p)
qq = twovar_marginals(q)
vcat([sum(abs, only(marginals(p)) - only(marginals(q)))],
    [sum(abs, pp[1,t] - qq[1,t]) for t in 2:p.L])