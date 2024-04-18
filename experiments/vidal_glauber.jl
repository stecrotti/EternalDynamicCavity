using MPSExperiments
using ProgressMeter, Tullio, TensorCast, LinearAlgebra
using TensorTrains, TensorTrains.UniformTensorTrains

d = 12
maxiter = 10
errs = fill(NaN, maxiter)
ovls = fill(NaN, maxiter)
εs = fill(NaN, maxiter)
bds = fill(NaN, maxiter)
beliefs = [complex.([NaN,NaN]) for _ in 1:maxiter]
A0 = reshape([0.4 0.4; 0.2 0.2], 1, 1, 2, 2)

J = 0.8
β = 1.0
h = 0.0
function f(J, h, β, xᵢᵗ⁺¹,xₙᵢᵗ,xᵢᵗ)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)
    potts2spin(x) = 3-2x
    hⱼᵢ = β*J * sum(potts2spin, xₙᵢᵗ; init=0.0)
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + β*h)
    return 1 / (1 + exp(2E))
end

function iterate_bp_vidal(f::Function, sz::Integer;
        maxiter=50, tol=1e-3,
        A0 = ones(1, 1, 2, 2) .+ 1e-5 .* rand.(),
        errs = fill(NaN, maxiter),
        ovls = fill(NaN, maxiter),
        εs = fill(NaN, maxiter),
        bds = fill(NaN, maxiter),
        beliefs = [complex.([NaN,NaN]) for _ in 1:maxiter])
    A = copy(A0)
    marg = fill(1/4, 4)
    @showprogress for it in 1:maxiter
        @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
            f(J,h,β,xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
        @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
        U, λ, V = svd(Matrix(Q))
        @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k:length(λ), xᵢᵗ:2, xⱼᵗ:2
        @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
        @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
        
        Mresh = reshape(M, size(M,1), size(M,2), :) .|> complex
        ψ = normalize!(iMPS(Mresh))
        p = InfiniteUniformTensorTrain(ψ)
        ϕ = canonicalize(ψ; svd_trunc=TruncBond(sz))
        bds[it] = MPSExperiments.bond_dims(ϕ)
        ovls[it] = overlap(ψ,ϕ) |> abs
        q = InfiniteUniformTensorTrain(ϕ)
        marg_new = marginals(q) |> only

        εs[it] = maximum(abs, marg - only(marginals(p)))
        errs[it] = maximum(abs, only(marginals(p)) - only(marginals(q)))
        marg = marg_new
        A = reshape(q.tensor, size(q.tensor, 1), size(q.tensor, 2), 2, 2)
        b = belief(A)
        beliefs[it] .= b ./ sum(b)
        εs[it] < tol && return A, maxiter, εs, errs, ovls, bds, beliefs
    end
    return A, maxiter, εs, errs, ovls, bds, beliefs
end

A, iters = iterate_bp_vidal(f, d; A0, tol=1e-6, maxiter, errs, ovls, εs, bds, beliefs)

ms = reduce.(-, beliefs |> real)