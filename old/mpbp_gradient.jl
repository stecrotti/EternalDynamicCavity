using MPSExperiments
using MatrixProductBP, MatrixProductBP.Models
using TensorTrains, Random, Tullio, TensorCast
using LinearAlgebra
using UnicodePlots
using ProgressMeter
import TensorTrains: UniformTensorTrain
using DiffResults, ForwardDiff
using Unzip
using Plots

function DifferenceNorm(q, p)
    sz = size(q.tensor)
    L = q.L

    function difference_norm(x)
        A = reshape(x, sz)
        q = UniformTensorTrain(A, L)
        return norm2m(q, p)
    end
end

function findmin_norm!(A, p; tol=1e-18, maxiter=100, η=1e-4, γ=1e-2,
        L = fill(NaN, maxiter+1))
    q = UniformTensorTrain(A, p.L)
    normalize!(q)
    for it in 1:maxiter
        foo = DifferenceNorm(q, p)
        res = DiffResults.GradientResult(A[:])
        ForwardDiff.gradient!(res, foo, A[:])
        L[it+1] = DiffResults.value(res)
        abs(L[it+1] - L[it]) < tol && return q, L
        dA = reshape(DiffResults.gradient(res), size(A))
        A .-= η * sign.(dA)
        η *= (1-γ)
        println("Iter $it of $maxiter: ℒ=$(L[it+1]), |∇L|=$(norm(dA)/length(dA))")
    end
    return q, L
end

function oneiter_A_finite!(w, A, T, maxiter_inner, L, tol_inner)
    @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
    w(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
    @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
    U, λ, V = svd(Matrix(Q))
    @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k:length(λ), xᵢᵗ:2, xⱼᵗ:2
    @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹:2
    @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
    p = UniformTensorTrain(M, T)
    normalize!(p)

    qnew, L = findmin_norm!(A, p; η=1e-3, γ=10 / maxiter_inner, maxiter=maxiter_inner, L,
        tol=tol_inner)
    m = only(marginals(qnew))
    d = norm(only(marginals(p)) - m)
    return qnew, m, d
end

function iterate_A_finite(w::HomogeneousGlauberFactor, sz::Integer; T=5,
        maxiter=50, tol=1e-3,
        maxiter_inner=200, tol_inner=1e-18,
        A0 = rand(rng, sz, sz, 2, 2))
    εs = fill(NaN, maxiter)
    ds = fill(NaN, maxiter)
    margq = [fill(1/4, 2, 2) for _ in 1:maxiter+1]
    L = fill(NaN, maxiter_inner+1)
    A = copy(A0)
    qnew = UniformTensorTrain(A, T)
    @showprogress for it in 1:maxiter
        qnew, margq[it+1], ds[it] = oneiter_A_finite!(w, A, T, maxiter_inner, L, tol_inner)
        εs[it] = sum(abs, margq[it] - margq[it+1])
        εs[it] < tol && return qnew, it, εs, ds, margq
        L .= NaN
    end
    return qnew, maxiter, εs, ds, margq
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

J = 0.2
β = 1.0
h = 0.2
w = HomogeneousGlauberFactor(J, h, β)
T = 6
maxiter = 10
maxiter_inner = 70
tol_inner = 1e-18
sz = 5

rng = MersenneTwister(0)
A = rand(rng, sz, sz, 2, 2)
Ls = [fill(NaN, maxiter_inner+1) for _ in 1:3]

qnew, m, d = map(Ls) do L
    oneiter_A_finite!(w, A, T, maxiter_inner, L, tol_inner)
end |> unzip

include("../../telegram/notifications.jl")
@telegram "ue"

# qnew, iters, εs, ds, margq = iterate_A_finite(w, sz; T, maxiter, maxiter_inner)

Ab = belief(A, w)
m_bp = reduce(-, only(marginals(UniformTensorTrain(Ab, T))))

import MatrixProductBP.Models: equilibrium_observables
m_eq, = equilibrium_observables(RandomRegular(3), J; β, h)
m_bp, m_eq