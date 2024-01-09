using MPSExperiments, TensorTrains, Random
import TensorTrains: UniformTensorTrain, marginals, InfiniteUniformTensorTrain
using Optim
using ForwardDiff


function NormDifference(q, p)
    sz = size(q.tensor)
    L = q.L

    function difference_norm(x)
        A = reshape(x, sz)
        q = UniformTensorTrain(A, L)
        return norm2m(q, p)
    end
end

rng = MersenneTwister(0)
D = 10
d = 7
M = rand(rng, D, D, 2, 3)
L = 5
p = UniformTensorTrain(M, L)
normalize!(p)

A0 = rand(rng, d, d, 2, 3)
q = UniformTensorTrain(A0, p.L)
normdiff = NormDifference(q, p)
res = optimize(normdiff, x -> ForwardDiff.gradient(normdiff, x), A0[:], 
    LBFGS(); inplace=false)

A = reshape(res.minimizer, size(A0))
q = UniformTensorTrain(A, L)

# using Plots
# display(plot(L, yaxis=:log10))

pp = twovar_marginals(p)
qq = twovar_marginals(q)
vcat([sum(abs, only(marginals(p)) - only(marginals(q)))],
    [sum(abs, pp[1,t] - qq[1,t]) for t in 2:p.L])