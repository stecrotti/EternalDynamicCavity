function truncate_vumps(A::Array, d; 
        ψ = InfiniteMPS([TensorMap(rand(d, size(A,2), d), (ℝ^d ⊗ ℝ^size(A,2)), ℝ^d)]),
        maxiter = 100, kw_vumps...)
    Q = size(A, 2)
    m = size(A, 1)
    @assert size(A, 3) == m
    t = TensorMap(A,(ℝ^m ⊗ ℝ^Q), ℝ^m) # the same but as a type digestible by MPSKit.jl
    ψ₀ = InfiniteMPS([t])
    II = DenseMPO([MPSKit.add_util_leg(id(storagetype(MPSKit.site_type(ψ₀)), physicalspace(ψ₀, i)))
        for i in 1:length(ψ₀)])
    alg = VUMPS(; maxiter, kw_vumps...) # variational approximation algorithm
    # alg = IDMRG1(; maxiter)
    @assert typeof(ψ) == typeof(ψ₀)
    ψ_, = approximate(ψ, (II, ψ₀), alg)   # do the truncation
    @assert typeof(ψ) == typeof(ψ_)

    ovl = abs(dot(ψ_, ψ₀))
    B = reshape(only(ψ_.AL).data, d, Q, d)
    return B, ovl, ψ_
end

function one_bpvumps_iter(f, A, sz, ψold, Aold, maxiter_vumps; kw_vumps...)
    @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
        f(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
    @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
    U, λ, V = svd(Matrix(Q))
    @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] (k in 1:length(λ), xᵢᵗ in 1:2, xⱼᵗ in 1:2)
    @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
    @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]

    Mresh = reshape(M, size(M,1), size(M,2), :)
    p = InfiniteUniformTensorTrain(Mresh)

    B = permutedims(Mresh, (1,3,2))
    Mtrunc, ovl, ψold = truncate_vumps(B, sz; ψ=ψold, maxiter=maxiter_vumps, kw_vumps...)
    Mtrunc ./= Mtrunc[1]
    Mtrunc_resh = permutedims(Mtrunc, (1,3,2))
    q = InfiniteUniformTensorTrain(Mtrunc_resh)

    err = maximum(abs, only(marginals(p)) - only(marginals(q)))
    A = reshape(Mtrunc_resh, size(Mtrunc_resh, 1), size(Mtrunc_resh, 2), 2, 2)
    ε = abs( 1 - dot(q, InfiniteUniformTensorTrain(Aold)))
    b = belief(A); b ./= sum(b)
    return A, ε, err, ovl, b
end


function iterate_bp_vumps(f::Function, sz::Integer;
        maxiter=50, tol=1e-3,
        A0 = reshape(rand(2,2), 1,1,2,2),
        maxiter_vumps = 100, kw_vumps...)
    errs = fill(NaN, maxiter)
    ovls = fill(NaN, maxiter)
    εs = fill(NaN, maxiter)
    beliefs = [[NaN,NaN] for _ in 1:maxiter]
    A = copy(A0)
    A0_expanded = zeros(sz,sz,2,2); A0_expanded[1:size(A0,1),1:size(A0,2),:,:] .= A0
    A0_expanded_reshaped = reshape(A0_expanded, size(A0_expanded,1), size(A0_expanded,2), :)
    t = permutedims(A0_expanded_reshaped, (1,3,2))
    ψold = InfiniteMPS([TensorMap(t, (ℝ^sz ⊗ ℝ^4), ℝ^sz)])
    As = [copy(A0)]
    prog = Progress(maxiter, desc="Running BP + VUMPS")
    for it in 1:maxiter
        Aold = As[end]
        A, ε, err, ovl, b = one_bpvumps_iter(f, A, sz, ψold, Aold, maxiter_vumps; kw_vumps...)
        push!(As, A)
        errs[it] = err
        beliefs[it] .= b
        ovls[it] = ovl
        εs[it] = ε
        εs[it] < tol && return A, maxiter, εs, errs, ovls, beliefs, As
        next!(prog, showvalues=[(:ε, "$(εs[it])/$tol")])
    end
    return A, maxiter, εs, errs, ovls, beliefs, As
end

#### BIPARTITE GRAPH

# return the average belief and the two block beliefs
function belief_bipartite(A, B)
    bijA = pair_belief(A)
    bijB = pair_belief(B)
    bA = sum(bijA, dims=2) |> vec
    bA ./= sum(bA)
    bB = sum(bijB, dims=2) |> vec
    bB ./= sum(bB)
    b = bA + bB
    b ./= sum(b)
    return b, bA, bB
end

# BP+VUMPS on bipartite graph with fixed degree k=3
function iterate_bp_vumps_bipartite(fA, fB, sz::Integer;
        maxiter=50, tol=1e-10,
        A0 = reshape(rand(2,2), 1,1,2,2),
        B0 = copy(A0),
        maxiter_vumps = 100, kw_vumps...)
    errs = fill(NaN, maxiter)
    ovls = fill(NaN, maxiter)
    εs = fill(NaN, maxiter)
    beliefs = [[NaN,NaN] for _ in 1:maxiter]
    beliefsA = [[NaN,NaN] for _ in 1:maxiter]; beliefsB = [[NaN,NaN] for _ in 1:maxiter]
    A = copy(A0); B = copy(B0)
    A0_expanded = zeros(sz,sz,2,2); A0_expanded[1:size(A0,1),1:size(A0,2),:,:] .= A0
    A0_expanded_reshaped = reshape(A0_expanded, size(A0_expanded,1), size(A0_expanded,2), :)
    tA = permutedims(A0_expanded_reshaped, (1,3,2))
    ψAold = InfiniteMPS([TensorMap(tA, (ℝ^sz ⊗ ℝ^4), ℝ^sz)])
    B0_expanded = zeros(sz,sz,2,2); B0_expanded[1:size(B0,1),1:size(B0,2),:,:] .= A0
    B0_expanded_reshaped = reshape(B0_expanded, size(B0_expanded,1), size(B0_expanded,2), :)
    tB = permutedims(B0_expanded_reshaped, (1,3,2))
    ψBold = InfiniteMPS([TensorMap(tB, (ℝ^sz ⊗ ℝ^4), ℝ^sz)])
    As = [copy(A0)]; Bs = [copy(B0)]
    prog = Progress(maxiter, desc="Running BP + VUMPS")
    for it in 1:maxiter
        Aold = As[end]; Bold = Bs[end]
        B, εB, errB, ovlB = one_bpvumps_iter(fB, A, sz, ψBold, Bold, maxiter_vumps; kw_vumps...)
        push!(Bs, B)
        A, εA, errA, ovlA = one_bpvumps_iter(fA, B, sz, ψAold, Aold, maxiter_vumps; kw_vumps...)
        push!(As, A)
        errs[it] = max(errA, errB)
        b, bA, bB = belief_bipartite(A, B)
        beliefs[it] = b; beliefsA[it] = bA; beliefsB[it] = bB
        ovls[it] = max(ovlA, ovlB)
        εs[it] = max(εA, εB)
        εs[it] < tol && return A, B, maxiter, εs, errs, ovls, beliefs, beliefsA, beliefsB, As, Bs
        next!(prog, showvalues=[(:ε, "$(εs[it])/$tol")])
    end
    return A, B, maxiter, εs, errs, ovls, beliefs, beliefsA, beliefsB, As, Bs
end