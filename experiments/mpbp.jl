using MPSExperiments
using MatrixProductBP, MatrixProductBP.Models
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using UnicodePlots

J = 0.5
β = 1.0
h = 0.2
w = HomogeneousGlauberFactor(J, h, β)

function iterate_A(w::HomogeneousGlauberFactor, sz::Integer;
        maxiter=50, tol=1e-3, damp=0.5,
        maxiter_inner=200, tol_inner=1e-5, damp_inner=0.8,
        A0 = ones(sz, sz, 2, 2))
    εs = fill(NaN, maxiter)
    A = copy(A0)
    for it in 1:maxiter
        @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
            w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
        @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
        U, λ, V = svd(Matrix(Q))
        @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k in 1:length(λ), xᵢᵗ in 1:2, xⱼᵗ in 1:2
        @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
        @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
        p = InfiniteUniformTensorTrain(M)
        p.tensor ./= sqrt(abs(tr(infinite_transfer_operator(p))))
        A0 = ones(sz, sz, size(p.tensor)[3:end]...)
        Anew = truncate_utt(p, sz; A0, maxiter=maxiter_inner, tol=tol_inner, damp=damp_inner)
        qnew = InfiniteUniformTensorTrain(Anew)
        Anew ./= sqrt(abs(tr(infinite_transfer_operator(qnew))))
        # εs[it] = norm(A - Anew) / sz
        εs[it] = norm(marginals(InfiniteUniformTensorTrain(A))[1] - marginals(qnew)[1])
        A .= damp * A + (1-damp) * Anew 
    end
    return A, maxiter, εs
end

function belief(A, w::HomogeneousGlauberFactor)
    @tullio BB[m1,m2,m3,n1,n2,n3,xᵢᵗ,xᵢᵗ⁺¹] := 
            w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ]*A[m3,n3,xⱼᵗ,xᵢᵗ] (xᵢᵗ⁺¹ in 1:2)
    @cast M[(xᵢᵗ,m1,m2,m3), (n1,n2,n3,xᵢᵗ⁺¹)] := BB[m1,m2,m3,n1,n2,n3,xᵢᵗ,xᵢᵗ⁺¹]
    U, λ, V = svd(Matrix(M))
    @cast C[m,k,xᵢᵗ] := U[(xᵢᵗ,m),k] k in 1:length(λ), xᵢᵗ in 1:2
    @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
    @tullio Anew[m,n,xᵢᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ]
    Anew
end

# sz = 5  
# A, maxiter, εs = iterate_A(w, sz; damp=0.0, maxiter=20, damp_inner=0, maxiter_inner=50)
# lineplot(εs, yscale=:log10) |> display
# q = InfiniteUniformTensorTrain(A)
# marginals(q)[1]

# b = belief(A, w)
# m = reduce(-, marginals(InfiniteUniformTensorTrain(b))[1])

sizes = 1:6
ms = map(sizes) do sz
    println("######\n\tSize $sz\n######")
    A, maxiter, εs = iterate_A(w, sz; damp=0.0, maxiter=20, damp_inner=0, maxiter_inner=50)
    lineplot(εs, yscale=:log10) |> display
    q = InfiniteUniformTensorTrain(A)
    marginals(q)[1]

    b = belief(A, w)
    m = reduce(-, marginals(InfiniteUniformTensorTrain(b))[1])
end

# sz = 3
# A = rand(sz, sz, 2, 2)
# @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
#     w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
# @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
# U, λ, V = svd(Matrix(Q))
# @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k in 1:length(λ), xᵢᵗ in 1:2, xⱼᵗ in 1:2
# @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
# @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
# p = InfiniteUniformTensorTrain(M)
# p.tensor ./= sqrt(abs(tr(infinite_transfer_operator(p))))
# marginals(p)

# @time begin
#     A = truncate_utt(p, sz; damp=0.8, maxiter=10^3, showprogress=true)
# end
# q = InfiniteUniformTensorTrain(A)
# d1 = norm(marginals(q)[1] - marginals(p)[1])

# @time begin
#     A = truncate_utt_eigen(p, sz; damp=0.8, maxiter=10^3, showprogress=true)
# end
# q = InfiniteUniformTensorTrain(A)
# d2 = norm(marginals(q)[1] - marginals(p)[1])

# d1, d2
