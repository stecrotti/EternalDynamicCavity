using ForwardDiff: gradient
using FiniteDifferences

rng = MersenneTwister(0)
L = 5
A = rand(rng, 2,2,3,4)
C = rand(rng, 3,3,3,4)
q = UniformTensorTrain(A, L)
p = UniformTensorTrain(C, L)
x = reshape(A, :)

function mydot(x)
    A = reshape(x, 2,2,3,4)
    q = UniformTensorTrain(A, L)
    dot(p, q)
end

g_forw = reshape(gradient(mydot, x), size(A))
g_fin = reshape(grad(central_fdm(5, 1), mydot, x)[1], size(A))
@test g_forw ≈ g_fin

G = transfer_operator(q, p)
@test mydot(x) ≈ tr(G^L)

# GL = prod(fill(G, L-1)).tensor
GL = collect((G^(L-1)))
@tullio g[a,b,x,y] := GL[b,j,a,l] * C[l,j,x,y] *($L)
@test g ≈ g_forw

function mynorm(x)
    A = reshape(x, 2,2,3,4)
    q = UniformTensorTrain(A, L)
    norm(q)^2
end

g_forw = reshape(gradient(mynorm, x), size(A))

E = transfer_operator(q)
@test mynorm(x) ≈ tr(E^L)

EL = collect((E^(L-1)))
# EL = prod(fill(E, L-1)).tensor
@tullio g[a,b,x,y] := EL[b,j,a,l] * A[l,j,x,y] * 2*($L)
@test g ≈ g_forw