using MPSExperiments
using MatrixProductBP, MatrixProductBP.Models
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using UnicodePlots
using ProgressMeter
import TensorTrains: UniformTensorTrain

J = 0.1
β = 1.0
h = 0.2
w = HomogeneousGlauberFactor(J, h, β)

function iterate_A(w::HomogeneousGlauberFactor, sz::Integer;
        maxiter=50, tol=1e-3, damp=0.5,
        maxiter_inner=200, tol_inner=1e-5,
        A0 = ones(sz, sz, 2, 2) .+ 1e-5 .* rand.())
    εs = fill(NaN, maxiter)
    ds = fill(NaN, maxiter)
    ovl = fill(NaN, maxiter)
    margq = [zeros(2,2) for _ in 1:maxiter]
    margp = [zeros(2,2) for _ in 1:maxiter]
    A = copy(A0)
    B = zeros(sz, sz, size(A0)[3:end]...)
    @showprogress for it in 1:maxiter
        @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
            w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
        @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
        U, λ, V = svd(Matrix(Q))
        @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k:length(λ), xᵢᵗ:2, xⱼᵗ:2
        @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹:2
        @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
        p = InfiniteUniformTensorTrain(M)
        # p.tensor ./= sqrt(abs(tr(infinite_transfer_operator(p))))
        normalize!(p)

        A0 = ones(sz, sz, size(p.tensor)[3:end]...)
        # qnew = truncate_variational(p, sz; A0, maxiter=maxiter_inner, tol=tol_inner)
        # Anew = qnew.tensor
        # qnew = InfiniteUniformTensorTrain(Anew)
        qnew = truncate_eachtensor(p, sz)
        Anew = qnew.tensor
        ovl[it] = dot(qnew, p)

        # C = _L(p) *_R(p)
        # U, λ, V = TruncBond(sz)(C)
        # for x in Iterators.product((1:q for q in size(M)[3:end])...)
        #     B[:,:,x...] .= U' * M[:,:,x...] * U
        # end
        # qnew = InfiniteUniformTensorTrain(B)
        # Anew = qnew.tensor
        # Anew ./= sqrt(abs(tr(infinite_transfer_operator(qnew))))
        ds[it] = norm(only(marginals(p)) - only(marginals(qnew)))
        # @show only(marginals(qnew))
        margq[it] = real(only(marginals(qnew)))
        margp[it] = marginals(p)[1]
        εs[it] = norm(marginals(InfiniteUniformTensorTrain(A))[1] - marginals(qnew)[1])
        A .= damp * A + (1-damp) * Anew 
    end
    return A, maxiter, εs, ds, ovl, margq, margp
end

function iterate_A_finite(w::HomogeneousGlauberFactor, sz::Integer, L::Integer;
        maxiter=50, tol=1e-3, damp=0.5,
        maxiter_inner=200, tol_inner=1e-5, damp_inner=0.8,
        A0 = ones(sz, sz, 2, 2) .+ 1e-5 .* rand.())
    εs = fill(NaN, maxiter)
    ds = fill(NaN, maxiter)
    margq = [zeros(2,2) for _ in 1:maxiter]
    margp = [zeros(2,2) for _ in 1:maxiter]
    A = copy(A0)
    # B = zeros(sz, sz, size(A0)[3:end]...)
    @showprogress for it in 1:maxiter
        @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
            w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
        @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
        U, λ, V = svd(Matrix(Q))
        @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k:length(λ), xᵢᵗ:2, xⱼᵗ:2
        @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹:2
        @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
        # @show sum(abs, M[:,:,1,1]) sum(abs, diag(M[:,:,1,1]))
        p = UniformTensorTrain(M, L)
        # p.tensor ./= sqrt(abs(tr(infinite_transfer_operator(p))))
        normalize!(p)

        # C = _L(p) *_R(p)
        # U, λ, V = TruncBond(sz)(C)
        # for x in Iterators.product((1:q for q in size(M)[3:end])...)
        #     B[:,:,x...] .= U' * M[:,:,x...] * U
        # end
        # qnew = UniformTensorTrain(B, L)
        @show size(M)
        qnew = truncate_eachtensor(p, sz)
        Anew = qnew.tensor
        @show size(Anew)
        # @show sum(abs, Anew[:,:,1,1]) sum(abs, diag(Anew[:,:,1,1]))
        # Anew ./= sqrt(abs(tr(transfer_operator(qnew))))
        ds[it] = norm(marginals(p)[1] - marginals(qnew)[1])
        margq[it] = marginals(qnew)[1]
        margp[it] = marginals(p)[1]
        εs[it] = norm(marginals(UniformTensorTrain(A, L))[1] - marginals(qnew)[1])
        A .= damp * A + (1-damp) * Anew 
    end
    return A, maxiter, εs, ds, margq, margp
end

function belief(A, w::HomogeneousGlauberFactor)
    @tullio BB[m1,m2,m3,n1,n2,n3,xᵢᵗ,xᵢᵗ⁺¹] := 
            w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ]*A[m3,n3,xⱼᵗ,xᵢᵗ] (xᵢᵗ⁺¹ in 1:2)
    @cast M[(xᵢᵗ,m1,m2,m3), (n1,n2,n3,xᵢᵗ⁺¹)] := BB[m1,m2,m3,n1,n2,n3,xᵢᵗ,xᵢᵗ⁺¹]
    U, λ, V = svd(Matrix(M))
    @cast C[m,k,xᵢᵗ] := U[(xᵢᵗ,m),k] k:length(λ), xᵢᵗ:2
    @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹:2
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

sizes = 1:4
# ms = map(sizes) do sz
    sz = 5
    # println("######\n\tSize $sz\n######")
    # A, maxiter, εs, ds, ovl, margq, margp = iterate_A(w, sz; damp=0.5, maxiter=30, 
    #     maxiter_inner=100)
    sz = 1; L = 2
    A, maxiter, εs, ds, margq, margp = iterate_A_finite(w, sz, L; maxiter=50, tol=1e-3)
    lineplot(εs, yscale=:log10, title ="Convergence error") |> display
    lineplot(ds, yscale=:log10, title = "Truncation error") |> display
    # lineplot(ovl, yscale=:log10, title = "Overlap") |> display
    # q = InfiniteUniformTensorTrain(A)
    # marginals(q)[1]

    b = belief(A, w)
    # m = reduce(-, marginals(InfiniteUniformTensorTrain(b))[1])
    m = reduce(-, marginals(UniformTensorTrain(b, L))[1])
    # end

# using Logging
# logger = SimpleLogger(stdout, Logging.Debug)
# with_logger(logger) do
    # sizes = 3:2:11
    # L = 50
    # ms = map(sizes) do sz
        # sz = 2
        # println("######\n\tSize $sz\n######")
        # A, maxiter, εs, ds, margq, margp = iterate_A_finite(w, sz, L; damp=0.0, maxiter=50, 
        #     damp_inner=0.0, maxiter_inner=40)
        # lineplot(εs, yscale=:log10, title ="Convergence error") |> display
        # lineplot(ds, yscale=:log10, title = "Truncation error") |> display
        # b = belief(A, w)
        # m = reduce(-, marginals(UniformTensorTrain(b, L))[1])
    # end

    # println(ms)
# end

import MatrixProductBP.Models: equilibrium_observables
m_eq, = equilibrium_observables(RandomRegular(3), J; β, h)
m, m_eq