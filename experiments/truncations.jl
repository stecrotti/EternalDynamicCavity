using MPSExperiments
using Test
using TensorTrains, Random, Tullio, TensorCast, ProgressMeter
using LinearAlgebra
using MatrixProductBP, MatrixProductBP.Models
using PrettyTables, Statistics

rng = MersenneTwister(1)
qs = (4)

L = 20
d = 10
sz = 3

N = 40
ds = map(1:N) do n
    println("Instance $n of $N")
    # M = rand(rng, d, d, qs...)
    # p = UniformTensorTrain(M, L)
    # normalize!(p)

    w = HomogeneousGlauberFactor(0.5, 0.2, 1.0)
    A = ones(sz, sz, 2, 2) .+ 1e-5 .* rand.()
    @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
    w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
    @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
    U, λ, V = svd(Matrix(Q))
    @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k in 1:length(λ), xᵢᵗ in 1:2, xⱼᵗ in 1:2
    @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
    @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
    p = UniformTensorTrain(M, L)
    normalize!(p)
    r = truncate_eachtensor(p, 10)
    marginals(p)[1] - marginals(r)[1]

    A = truncate_utt(p, sz; damp=0.8)
    q = UniformTensorTrain(A, L)
    d1 = norm(marginals(q)[1] - marginals(p)[1])

    p2 = deepcopy(p)
    orthogonalize_left!(p2)
    # @assert isapprox(marginals(p2), marginals(p), rtol=1e-12)
    A2 = truncate_utt(p2, sz; damp=0.8)
    q2 = UniformTensorTrain(A2, L)
    d2 = norm(marginals(q2)[1] - marginals(p)[1])

    r = truncate_eachtensor(p, sz)
    d3 = norm(marginals(p)[1] - marginals(r)[1])

    C = _L(p) *_R(p)
    U, λ, V = TruncBond(sz)(C)
    B = zeros(sz, sz, size(M)[3:end]...)
    for x in Iterators.product((1:q for q in size(M)[3:end])...)
        B[:,:,x...] .= U' * M[:,:,x...] * U
    end
    s = UniformTensorTrain(B, length(p))
    d4 = norm(marginals(p)[1] - marginals(s)[1])

    # pretty_table(["Error on marginals" d1 d2 d3 d4], header=["", "Variational", 
    #     "Orthogonalize, then variational",
    #     "Truncate each tensor", "Orthogonalize, then trunc each tensor"])
    [d1, d2, d3, d4]
end

avg = mean(ds)
sd = std(ds) ./ sqrt(N)

t = pretty_table(hcat(["Variational", 
"Orthogonalize, then variational",
"Truncate each tensor", "Orthogonalize, then trunc each tensor"], 
    ["$a ± $b" for (a,b) in zip(avg,sd)]),
header=["Truncation method", "Error on marginals - L=$L"])


N_inf = 40
ds_inf = map(1:N_inf) do n
    println("Instance $n of $N")
    # M = rand(rng, d, d, qs...)
    # p = UniformTensorTrain(M, L)
    # normalize!(p)

    w = HomogeneousGlauberFactor(0.5, 0.2, 1.0)
    A = rand(sz, sz, 2, 2)
    @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
    w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
    @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
    U, λ, V = svd(Matrix(Q))
    @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k in 1:length(λ), xᵢᵗ in 1:2, xⱼᵗ in 1:2
    @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
    @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
    p = InfiniteUniformTensorTrain(M)
    normalize!(p)
    r = truncate_eachtensor(p, 10)
    marginals(p)[1] - marginals(r)[1]

    A = truncate_utt(p, sz; damp=0.8)
    q = InfiniteUniformTensorTrain(A)
    d1 = norm(marginals(q)[1] - marginals(p)[1])

    p2 = deepcopy(p)
    orthogonalize_left!(p2)
    # @assert isapprox(marginals(p2), marginals(p), rtol=1e-12)
    A2 = truncate_utt(p2, sz; damp=0.8)
    q2 = InfiniteUniformTensorTrain(A2)
    d2 = norm(marginals(q2)[1] - marginals(p)[1])

    r = truncate_eachtensor(p, sz)
    d3 = norm(marginals(p)[1] - marginals(r)[1])

    C = _L(p) *_R(p)
    U, λ, V = TruncBond(sz)(C)
    B = zeros(sz, sz, size(M)[3:end]...)
    for x in Iterators.product((1:q for q in size(M)[3:end])...)
        B[:,:,x...] .= U' * M[:,:,x...] * U
    end
    s = InfiniteUniformTensorTrain(B)
    d4 = norm(marginals(p)[1] - marginals(s)[1])

    # pretty_table(["Error on marginals" d1 d2 d3 d4], header=["", "Variational", 
    #     "Orthogonalize, then variational",
    #     "Truncate each tensor", "Orthogonalize, then trunc each tensor"])
    [d1, d2, d3, d4]
end

avg_inf = mean(ds_inf)
sd_inf = std(ds_inf) ./ sqrt(N_inf)

t_inf = pretty_table(hcat(["Variational", 
"Orthogonalize, then variational",
"Truncate each tensor", "Orthogonalize, then trunc each tensor"], 
    ["$a ± $b" for (a,b) in zip(avg_inf,sd_inf)]),
header=["Truncation method", "Error on marginals - L=Inf"])

