rng = MersenneTwister(0)
L = 6
A = rand(rng, 2,2,3,4)
M = rand(rng, 3,3,3,4)
q = UniformTensorTrain(A, L)
p = UniformTensorTrain(M, L)

G = transfer_operator(q, p)

(; l, r, λ) = leading_eig(G)
@test l * G ≈ l * λ
@test G * r ≈ λ * r

qq = deepcopy(q)
E = transfer_operator(qq)
η = leading_eig(E)[:λ]
qq.tensor ./= √η
Einf = infinite_transfer_operator(E)
ks = 1:5
diffs = map(ks) do k
    Ek = E^k
    norm(collect(Ek) - abs.(collect(Einf)))
end
@test issorted(diffs, rev=true)

G = transfer_operator(p, q)
@test tr(G^L) ≈ dot(p, q)

E = transfer_operator(q)
@test tr(E^L) ≈ norm(q)^2