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
Einf = infinite_transfer_operator(transfer_operator(qq))
ks = 1:5

diffs = map(ks) do k
    Ek = E^k
    norm(collect(Ek) - real(collect(Einf))) / norm(collect(Ek))
end
@test issorted(diffs, rev=true)

G = transfer_operator(p, q)
@test tr(G^L) ≈ dot(p, q)

# function mypow(G::AbstractTransferOperator, k::Integer)
#     Gk = G
#     for _ in 1:k-1
#         Gk = Gk * G
#     end
#     return Gk
# end

mypow(G::AbstractTransferOperator, k::Integer) = prod(fill(G, k))

for k in ks
    Gk = G^k
    @test collect(mypow(G, k)) ≈ collect(Gk)
    @test tr(mypow(G, k)) ≈ tr(Gk)
end

E = transfer_operator(q)
@test tr(E^L) ≈ norm(q)^2