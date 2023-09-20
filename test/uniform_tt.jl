rng = MersenneTwister(0)
L = 10
q = UniformTensorTrain(rand(rng, 2,2,3,4), L)
p = UniformTensorTrain(rand(rng, 2,2,3,4), L)

G = transfer_operator(q, p)

(; l, r, λ) = eig(G)
@test l * G ≈ l * λ
@test G * r ≈ λ * r

q.tensor ./= sqrt(λ)
p.tensor ./= sqrt(λ)
G = transfer_operator(q, p)

G = transfer_operator(q)

# Ginf = infinite_power(G; e = (; l, r, λ))
# ks = 1:20
# diffs = map(ks) do k
#     Gk = G^k
#     norm(Gk - abs.(Ginf.tensor))
# end

# using UnicodePlots
# lineplot(ks, diffs, xlabel="k", ylabel="err", yscale=:log10)

# @test tr(G^L) ≈ dot(p, q)

# E = transfer_operator(q)
# @test tr(E^L) ≈ norm(q)

# using BenchmarkTools, ProgressMeter
# ks = 1:15
# map(ks) do k
#     @test (G^k).tensor ≈ slowpow(G, k).tensor
# end


