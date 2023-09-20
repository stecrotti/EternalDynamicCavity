using ForwardDiff: gradient

rng = MersenneTwister(0)
L = 10
B = rand(rng, 2,2,3,4)
q = UniformTensorTrain(B, L)
x = reshape(B, :)

function mynorm(x)
    A = reshape(x, 2,2,3,4)
    q = UniformTensorTrain(A, L)
    n = norm(q)
end

grad = reshape(gradient(mynorm, x), size(A))

E = transfer_operator(q)
EL = E^(L-1)
# @tullio g[a,b,x,y] := 