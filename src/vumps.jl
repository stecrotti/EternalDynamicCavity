function truncate_vumps(A::Array, d; ψ = InfiniteMPS([TensorMap(rand(d, size(A,2), d), (ℝ^d ⊗ ℝ^size(A,2)), ℝ^d)]))
    Q = size(A, 2)
    m = size(A, 1)
    @assert size(A, 3) == m
    t = TensorMap(A,(ℝ^m ⊗ ℝ^Q), ℝ^m) # the same but as a type digestible by MPSKit.jl
    ψ₀ = InfiniteMPS([t])
    II = DenseMPO([MPSKit.add_util_leg(id(storagetype(MPSKit.site_type(ψ₀)), physicalspace(ψ₀, i)))
        for i in 1:length(ψ₀)])
    alg = VUMPS(; maxiter=100) # variational approximation algorithm
    # alg = IDMRG1(; maxiter=100)
    @assert typeof(ψ) == typeof(ψ₀)
    ψ_, = approximate(ψ, (II, ψ₀), alg)   # do the truncation
    @assert typeof(ψ) == typeof(ψ_)

    ovl = abs(dot(ψ_, ψ₀))
    B = reshape(only(ψ_.AL).data, d, Q, d)
    return B, ovl, ψ_
end


function iterate_bp_vumps(f::Function, sz::Integer;
        maxiter=50, tol=1e-3,
        A0 = reshape(rand(2,2), 1,1,2,2))
    errs = fill(NaN, maxiter)
    ovls = fill(NaN, maxiter)
    εs = fill(NaN, maxiter)
    beliefs = [[NaN,NaN] for _ in 1:maxiter]
    A = copy(A0)
    marg = fill(1/4, 4)
    A0_expanded = zeros(sz,sz,2,2); A0_expanded[1:1,1:1,:,:] .= A0
    A0_expanded_reshaped = reshape(A0_expanded, size(A0_expanded,1), size(A0_expanded,2), :)
    t = permutedims(A0_expanded_reshaped, (1,3,2))
    ψold = InfiniteMPS([TensorMap(t, (ℝ^sz ⊗ ℝ^4), ℝ^sz)])
    As = [copy(A0)]
    @showprogress for it in 1:maxiter
        @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
            f(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
        @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
        U, λ, V = svd(Matrix(Q))
        @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] (k in 1:length(λ), xᵢᵗ in 1:2, xⱼᵗ in 1:2)
        @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
        @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]
        
        Mresh = reshape(M, size(M,1), size(M,2), :)
        p = InfiniteUniformTensorTrain(Mresh)
        λ, = TensorTrains.UniformTensorTrains._eigen(p)
        Mresh ./= exp(im*angle(λ))
    
        B = permutedims(Mresh, (1,3,2))
        Mtrunc, ovls[it], ψold = truncate_vumps(B, sz; ψ=ψold)
        Mtrunc ./= Mtrunc[1]
        Mtrunc_resh = permutedims(Mtrunc, (1,3,2))
        q = InfiniteUniformTensorTrain(Mtrunc_resh)
        marg_new = marginals(q) |> only

        # εs[it] = maximum(abs, marg - only(marginals(p)))
        errs[it] = maximum(abs, only(marginals(p)) - only(marginals(q)))
        marg = marg_new
        A = reshape(Mtrunc_resh, size(Mtrunc_resh, 1), size(Mtrunc_resh, 2), 2, 2)
        push!(As, A)
        εs[it] = abs( 1 - dot(q, InfiniteUniformTensorTrain(As[end-1])))
        b = belief(A)
        beliefs[it] .= b ./ sum(b)
        εs[it] < tol && return A, maxiter, εs, errs, ovls, beliefs, As
    end
    return A, maxiter, εs, errs, ovls, beliefs, As
end


function pair_belief(A)
    @cast _[(aᵗ,bᵗ),(aᵗ⁺¹,bᵗ⁺¹),xᵢᵗ,xⱼᵗ] := A[aᵗ,aᵗ⁺¹,xᵢᵗ, xⱼᵗ] * A[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ]
end
belief(A) = sum(pair_belief(A), dims=(1,2,3)) |> vec