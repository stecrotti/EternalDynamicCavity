using MatrixProductBP, MatrixProductBP.Models
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra

function iterate_A(w::HomogeneousGlauberFactor, sz::Integer, L::Integer;
        maxiter=10, tol=1e-3, damp=0.0,
        maxiter_inner=200, tol_inner=1e-5, damp_inner=0.8)
    εs = fill(NaN, maxiter)
    ms = fill(NaN, maxiter)
    A = rand(sz, sz, 2, 2)
    for it in 1:maxiter
        @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
            w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
        @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
        U, λ, V = svd(Matrix(Q))
        @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k in 1:length(λ), xᵢᵗ in 1:2, xⱼᵗ in 1:2
        @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
        @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
        p = UniformTensorTrain(M, L)
        p.tensor ./= norm(p)
        Anew = truncate_utt(p, sz; maxiter=maxiter_inner, tol=tol_inner, damp=damp_inner)
        εs[it] = norm(A - Anew) / sz
        A .= damp * A + (1-damp) * Anew 
    end
    return A, maxiter, εs
end

J = 0.5
β = 1.0
h = 0.0
w = HomogeneousGlauberFactor(J, h, β)

sz = 2
L = 5
A, maxiter, εs = iterate_A(w, sz, L, maxiter=100)
εs