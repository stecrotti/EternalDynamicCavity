using ForwardDiff: gradient
using FiniteDifferences

rng = MersenneTwister(0)
L = 10
A = rand(rng, 2,2,12)
M = rand(rng, 3,3,12)
q = UniformTensorTrain(A, L)
p = UniformTensorTrain(M, L)
x = reshape(A, :)

function mydot(x)
    A = reshape(x, 2,2,12)
    q = UniformTensorTrain(A, L)
    dot(p, q)
end

g_forw = reshape(gradient(mydot, x), size(A))
g_fin = reshape(grad(central_fdm(5, 1), mydot, x)[1], size(A))
@test g_forw ≈ g_fin

G = transfer_operator(p, q)
@test mydot(x) ≈ tr(G^L)

g = gradientA(p, q)
@test g ≈ g_forw

function mynorm(x)
    A = reshape(x, 2,2,12)
    q = UniformTensorTrain(A, L)
    norm(q)^2
end

g_forw = reshape(gradient(mynorm, x), size(A))

E = transfer_operator(q)
@test mynorm(x) ≈ tr(E^L)

g = gradientA(q)
@test g ≈ g_forw