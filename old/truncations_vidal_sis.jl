using Tullio
using KrylovKit: eigsolve
using LinearAlgebra
using ProgressMeter
using Random, TensorCast
using Plots, LaTeXStrings
using Base.Threads


struct iMPS{T<:Number,F<:Number}
    Γ :: Array{T, 3}
    λ :: Vector{F}
end
function iMPS(A::Array{T,3}) where {T<:Number}
    d = size(A, 1)
    @assert d == size(A, 2)
    return iMPS(A, ones(d)) 
end

bond_dims(ψ::iMPS) = size(ψ.Γ, 1)

function LinearAlgebra.dot(ψ::iMPS, ϕ::iMPS)
    (; Γ, λ) = ψ
    Γp, λp = ϕ.Γ, ϕ.λ
    @assert size(Γ, 3) == size(Γp, 3)
    
    matrixify(v) = reshape(v, size(Γ, 2), size(Γp, 2))
    function fR(v)
        V = matrixify(v)
        @tullio tmp[ap,b,i] := conj(Γp[ap,bp,i] * λp[bp]) * V[b,bp]
        (@tullio _[a,ap] := Γ[a,b,i] * λ[b] * tmp[ap,b,i]) |> vec
    end
    valsR, vecsR, infoR = eigsolve(fR, vec(Matrix(1.0I,size(Γ, 2),size(Γp,2))))
    η = valsR[1]
end
overlap(ψ::iMPS, ϕ::iMPS) = dot(ψ, ϕ) / sqrt(dot(ψ,ψ) * dot(ϕ,ϕ))
LinearAlgebra.norm(ψ::iMPS) = sqrt(dot(ψ,ψ))
LinearAlgebra.normalize!(ψ::iMPS) = (ψ.Γ ./= norm(ψ); return ψ)        

using TensorTrains, TensorTrains.UniformTensorTrains
function TensorTrains.UniformTensorTrains.InfiniteUniformTensorTrain(ψ::iMPS)
   (; Γ, λ) = ψ
   @tullio A[a,b,i] := Γ[a,b,i] * λ[b]
    InfiniteUniformTensorTrain(A)
end

function canonicalize(ψ::iMPS; svd_trunc=svd)
    (; Γ, λ) = ψ
    d = length(λ)
    matrixify(v) = reshape(v, d, d)
    function fR(v)
        V = matrixify(v)
        # @tullio tmp[ap,b,i] := conj(Γ[ap,bp,i] * λ[bp]) * V[b,bp]
        # (@tullio _[a,ap] := Γ[a,b,i] * λ[b] * tmp[ap,b,i]) |> vec
        @tullio tmp[a,bp,i] := Γ[a,b,i] * λ[b] * V[b,bp]
        @tullio _[a,ap] := conj(Γ[ap,bp,i] * λ[bp]) * tmp[a,bp,i]
    end
    function fL(v)
        V = matrixify(v)
        @tullio tmp[a,bp,i] := conj(Γ[ap,bp,i] * λ[ap]) * V[a,ap]
        (@tullio _[b,bp] := Γ[a,b,i] * λ[a] * tmp[a,bp,i]) |> vec
    end
    valsR, vecsR, infoR = eigsolve(fR, vec(Matrix(1.0I,d,d)))
    valsL, vecsL, infoL = eigsolve(fL, vec(Matrix(1.0I,d,d)))
    valsR[1] ≈ valsL[1] || @warn "L and R do not have same leading eigenvalue: got $(valsR[1]), $(valsL[1])"
    VR = real(vecsR[1] ./ (vecsR[1][1] / abs(vecsR[1][1]))) |> matrixify |> Hermitian
    VL = real(vecsL[1] ./ (vecsL[1][1] / abs(vecsL[1][1]))) |> matrixify |> Hermitian
    U, S, V = svd(VR)
    X = U * Diagonal(sqrt.(S))
    U, S, V = svd(VL)
    Y = Diagonal(sqrt.(S)) * V'
    U, λp, V = svd_trunc(Y' * Diagonal(λ) * X)
    L = V' * inv(X)
    R = inv(Y') * U
    @tullio tmp[a,c,i] := L[a,b] * Γ[b,c,i]
    @tullio Γp[a,d,i] := tmp[a,c,i] * R[c,d]
    return iMPS(Γp, λp)
end

const SUSCEPTIBLE = 1 
const INFECTIOUS = 2

λ_ = 0.2        # rate of transmission
ρ_ = 0.1        # rate of recovery

function f(λ, ρ, xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)
    @assert xᵢᵗ⁺¹ ∈ 1:2
    @assert all(x ∈ 1:2 for x in xₙᵢᵗ)

    if xᵢᵗ == INFECTIOUS
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return ρ
        else
            return 1 - ρ 
        end
    else
        p = (1-λ)^sum(xⱼᵗ == INFECTIOUS for xⱼᵗ in xₙᵢᵗ; init=0.0)
        if xᵢᵗ⁺¹ == SUSCEPTIBLE
            return p
        elseif xᵢᵗ⁺¹ == INFECTIOUS
            return 1 - p
        end
    end
end

function iterate_bp_vidal!(A, f::Function, sz::Integer;
    maxiter=50, tol=1e-3,
    errs = fill(NaN, maxiter),
    ovls = fill(NaN, maxiter),
    εs = fill(NaN, maxiter),
    bds = fill(NaN, maxiter),
    beliefs = [complex.([NaN,NaN]) for _ in 1:maxiter])

    marg = fill(1/4, 4)

@showprogress for it in 1:maxiter
    @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
        f(λ_, ρ_, xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
    @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
    U, λ, V = svd(Matrix(Q))
    @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k:length(λ), xᵢᵗ:2, xⱼᵗ:2
    @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹:2
    @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
    
    Mresh = reshape(M, size(M,1), size(M,2), :) .|> complex
    ψ = normalize!(iMPS(Mresh))
    p = InfiniteUniformTensorTrain(ψ)
    ϕ = canonicalize(ψ; svd_trunc=TruncBond(sz))
    bds[it] = bond_dims(ϕ)
    ovls[it] = overlap(ψ,ϕ) |> abs
    q = InfiniteUniformTensorTrain(ϕ)
    marg_new = marginals(q) |> only

    εs[it] = maximum(abs, marg - only(marginals(p)))
    errs[it] = maximum(abs, only(marginals(p)) - only(marginals(q)))
    marg = marg_new
    A = reshape(q.tensor, size(q.tensor, 1), size(q.tensor, 2), 2, 2)
    b = belief(A)
    beliefs[it] .= b ./ sum(b)
    εs[it] < tol && return A, it, εs, errs, ovls, bds, beliefs
end
return A, maxiter, εs, errs, ovls, bds, beliefs
end

function pair_belief(A)
@cast _[(aᵗ,bᵗ),(aᵗ⁺¹,bᵗ⁺¹),xᵢᵗ,xⱼᵗ] := A[aᵗ,aᵗ⁺¹,xᵢᵗ, xⱼᵗ] * A[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ]
end

belief(A) = sum(pair_belief(A), dims=(1,2,3)) |> vec

# d = 8
# maxiter = 10
# errs = fill(NaN, maxiter)
# ovls = fill(NaN, maxiter)
# εs = fill(NaN, maxiter)
# bds = fill(NaN, maxiter)
# beliefs = [complex.([NaN,NaN]) for _ in 1:maxiter]
# A = reshape([0.111609  0.111609;  0.388391  0.388391], 1,1,2,2)

# A, iters = iterate_bp_vidal!(A, f, d; tol=1e-6, maxiter, errs, ovls, εs, bds, beliefs)

ds = 2:3:13
maxiter = 50

pls = map(eachindex(ds)) do a
    d = ds[a]
    errs = fill(NaN, maxiter)
    ovls = fill(NaN, maxiter)
    εs = fill(NaN, maxiter)
    bds = fill(NaN, maxiter)
    beliefs = [complex.([NaN,NaN]) for _ in 1:maxiter]
    A = reshape([0.111609  0.111609;  0.388391  0.388391], 1,1,2,2)
    A, iters = iterate_bp_vidal!(A, f, d; tol=1e-6, maxiter, errs, ovls, εs, bds, beliefs)
    p1 = plot(εs, xlabel="iter", ylabel="convergence error", yscale=:log10, title="d=$d")
    p2 = plot(errs, xlabel="iter", ylabel="truncation error on marginals", yscale=:log10)
    p3 = plot(abs.(1 .- ovls), xlabel="iter", ylabel="|1-truncation overlap|", yscale=:log10)
    p4 = plot(reduce.(-, beliefs) .|> real, ylabel="Re[single-var magnetiz]", ylims=(-2,2))
    pl = plot(p1, p2, p3, p4, layout=(1,4), size=(900,250), margin=5Plots.mm, labelfontsize=9,
        xlabel="iter", label="")
end

pl = plot(pls...; xlabel="iter", layout=(length(ds),1), size=(1000,1000))