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


function iterate_bp_vumps_mpskit(f::Function, sz::Integer;
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

function is_leftorth(AL)
    @tullio x[j,k] := AL[i,j,x] * AL[i,k,x]
    x ≈ I
end
function is_rightorth(AR)
    @tullio y[j,k] := AR[j,l,x] * AR[k,l,x]
    y ≈ I
end

function polar_left(A; kw...)
    U, = MatrixFactorizations.polar(A; kw...)
    U
end
function polar_right(A; kw...)
    U, = MatrixFactorizations.polar(collect(A'); kw...)
    U'
end

function qrpos_left(A::AbstractMatrix)
    Q_, R = qr(A)
    Q = Matrix(Q_)
    d = @view R[diagind(R)]
    if all(>(0), d)
        return Q
    else
        D = Diagonal(sign.(d))
        return Q * D
    end
end
qrpos_right(A::AbstractMatrix) = qrpos_left(A')'

function qrpos(A::AbstractMatrix)
    Q_, R = qr(A)
    Q = Matrix(Q_)
    d = @view R[diagind(R)]
    if all(>(0), d)
        return Q, R
    else
        D = Diagonal(sign.(d))
        Q = Q * D
        R = D * R
        return Q, R
    end
end

function qrpos_left(L::AbstractMatrix{T}, A::Array{T,3}) where T
    @tullio LxA[i,j,x] := L[i,k] * A[k,j,x]
    @cast LxA_[(i,x),j] := LxA[i,j,x]
    AL_, Lnew = qrpos(LxA_)
    @cast AL[i,j,x] := AL_[(i,x),j] (x in 1:size(A,3))
    return AL, Lnew
end
function qrpos_right(A::Array{T,3}, R::AbstractMatrix{T}) where T
    @tullio AxR[i,j,x] := A[i,k,x] * R[k,j] 
    @cast AxR_[i,(j,x)] := AxR[i,j,x]
    X, Y = qrpos(AxR_')
    Rnew, AR_ = Y', X'
    @cast AR[i,j,x] := AR_[i,(j,x)] (x in 1:size(A,3))
    return AR, Rnew
end

function cholesky_psd(A)
    F = cholesky(A, RowMaximum())
    L = F.L[invperm(F.p), 1:F.rank]
    U = F.U[1:F.rank, invperm(F.p)]
    return L, U
end

function _initialize_posdef(n; rng=Random.default_rng())
    l = rand(rng, n, n)
    l = l'l + I
    l ./= norm(l)
    return l
end

function left_orthogonal(A; l = _initialize_posdef(size(A,1)), 
        maxiter=2*10^3, tol=1e-16)
    ε = λl = Inf
    for _ in 1:maxiter
        @views lnew = sum(A[:,:,x]' * l * A[:,:,x] for x in axes(A,3))
        λl = norm(lnew)
        lnew ./= λl
        ε = norm(l - lnew)
        ε < tol && return lnew, λl
        l = lnew
    end
    @info "LeftOrthogonal not converged after $maxiter iterations. ε=$ε"
    return l, λl
end

function right_orthogonal(A; r = _initialize_posdef(size(A,1)), 
        maxiter=2*10^3, tol=1e-16)
    ε = λr = Inf
    for _ in 1:maxiter
        @views rnew = sum(A[:,:,x] * r * A[:,:,x]' for x in axes(A,3))
        λr = norm(rnew)
        rnew ./= λr
        ε = norm(r - rnew)
        ε < tol && return rnew, λr
        r = rnew
    end
    @info "RightOrthogonal not converged after $maxiter iterations. ε=$ε"
    return r, λr
end

function left_orthogonal_qr!(L, A;
        maxiter=2*10^3, tol=1e-16)
    ε = λL = Inf
    AL = copy(A)
    for _ in 1:maxiter
        AL, Lnew = qrpos_left(L, A)
        λL = norm(Lnew)
        Lnew ./= λL
        ε = norm(L - Lnew)
        L .= Lnew
        ε < tol && return AL, λL
    end
    @info "LeftOrthogonal not converged after $maxiter iterations. ε=$ε"
    return AL, λL
end
function right_orthogonal_qr!(R, A;
        maxiter=2*10^3, tol=1e-16)
    ε = λR = Inf
    AR = copy(A)
    for _ in 1:maxiter
        AR, Rnew = qrpos_right(A, R)
        λR = norm(Rnew)
        Rnew ./= λR
        ε = norm(R - Rnew)
        R .= Rnew
        ε < tol && return AR, λR
    end
    @info "RightOrthogonal not converged after $maxiter iterations. ε=$ε"
    return AR, λR
end

function mixed_canonical_qr!(L, R, A; maxiter_ortho=10^3, tol_ortho=1e-15)
    AL, λL = left_orthogonal_qr!(L, A; tol=tol_ortho, maxiter=maxiter_ortho)
    AR, λR = right_orthogonal_qr!(R, A; tol=tol_ortho, maxiter=maxiter_ortho)
    @debug @assert λL ≈ λR
    # @show λL
    # @show maximum(norm, AL[:,:,x]*L - L*A[:,:,x] for x in 1:4)

    # compute C, AC
    U, c, V = svd(L * R)
    Ctilde = Diagonal(c)
    @debug (@assert U * Ctilde * V' ≈ L * R)
    ALtilde = similar(A); ARtilde = similar(A)
    for x in axes(A,3)
        @views ALtilde[:,:,x] .= U' * AL[:,:,x] * U
        @views ARtilde[:,:,x] .= V' * AR[:,:,x] * V
    end
    ACtilde = similar(A)
    for x in axes(ACtilde,3)
        @views ACtilde[:,:,x] .= ALtilde[:,:,x] * Ctilde
    end
    @debug begin
        ACtilde2 = similar(A)
        for x in axes(A,3)
            @views ACtilde2[:,:,x] .= Ctilde * ARtilde[:,:,x]
        end
        @assert ACtilde2 ≈ ACtilde
    end
    @debug begin
        @assert is_leftorth(ALtilde)
        @assert is_rightorth(ARtilde)
        qL = InfiniteUniformTensorTrain(ALtilde)
        qR = InfiniteUniformTensorTrain(ARtilde)
        q = InfiniteUniformTensorTrain(A)
        @assert marginals(qL) ≈ marginals(qR) ≈ marginals(q)
    end
    return ALtilde, ARtilde, ACtilde, Ctilde
end

function mixed_canonical_original(A; maxiter_ortho=2*10^3)
    m = size(A, 1); q = size(A, 3)
    ψ = InfiniteMPS([TensorMap(permutedims(A, (1,3,2)), (ℝ^m ⊗ ℝ^q), ℝ^m)]; maxiter=maxiter_ortho)
    ALtilde = permutedims(reshape(ψ.AL[1].data, (m,q,m)), (1,3,2))
    ARtilde = permutedims(reshape(ψ.AR[1].data, (m,q,m)), (1,3,2))
    ACtilde = permutedims(reshape(ψ.AC[1].data, (m,q,m)), (1,3,2))
    Ctilde = ψ.CR[1].data
    return ALtilde, ARtilde, ACtilde, Ctilde
end

function left_fixedpoint(ALtilde, AL; L1 = rand(d, size(ALtilde,2)), maxiter=2*10^3, tol=1e-16)
    left_fixedpoint!(L1, ALtilde, AL; maxiter, tol)
    return L1
end
function left_fixedpoint!(L, ALtilde, AL; maxiter=2*10^3, tol=1e-16)
    ε = Inf
    Lold = copy(L); Lnew = copy(L)
    for _ in 1:maxiter
        @views Lnew .= sum(AL[:,:,x]' * Lold * ALtilde[:,:,x] for x in axes(ALtilde,3))
        Lnew ./= norm(Lnew)
        ε = norm(Lnew - Lold)
        if ε < tol
            L .= Lnew
            return nothing
        end
        Lold = Lnew
    end
    @info "left fixedpoint not converged after $maxiter iterations. ε=$ε"
    L .= Lnew
    return nothing
end
function right_fixedpoint(ARtilde, AR; R1 = rand(size(ARtilde,1), d), maxiter=2*10^3, tol=1e-16)
    right_fixedpoint!(R1, ARtilde, AR; maxiter, tol)
    return R1
end
function right_fixedpoint!(R, ARtilde, AR; maxiter=2*10^3, tol=1e-16)
    ε = Inf
    Rold = copy(R); Rnew = copy(R)
    for _ in 1:maxiter
        @views Rnew .= sum(ARtilde[:,:,x] * Rold * AR[:,:,x]' for x in axes(ARtilde,3))
        Rnew ./= norm(Rnew)
        ε = norm(Rnew - Rold)
        if ε < tol
            R .= Rnew
            return nothing
        end
        Rold = Rnew
    end
    @info "right fixedpoint not converged after $maxiter iterations. ε=$ε"
    R .= Rnew
    return nothing
end

function mixed_canonical(A; l = _initialize_posdef(size(A,1)), r = _initialize_posdef(size(A,1)),
        maxiter_ortho = 10^3, tol_ortho=1e-15)
    ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical!(l, r, A; maxiter_ortho, tol_ortho)
    return ALtilde, ARtilde, ACtilde, Ctilde
end

function mixed_canonical!(l, r, A; maxiter_ortho = 10^3, tol_ortho=1e-15)
    lnew, λl = left_orthogonal(A; l, tol=tol_ortho, maxiter=maxiter_ortho)
    rnew, λr = right_orthogonal(A; r, tol=tol_ortho, maxiter=maxiter_ortho)
    @debug @assert λl ≈ λr
    λ = tr(lnew * rnew)
    lnew ./= sqrt(λ); rnew ./= sqrt(λ); A ./= sqrt(λl)
    l .= lnew; r .= rnew
    # @show eigvals(lnew), eigvals(Hermitian(lnew)), size(lnew, 1)
    L = cholesky(Hermitian(lnew)).U
    # L = cholesky_psd(Hermitian(lnew))[2]
    Linv = pinv(L)
    R = cholesky(Hermitian(rnew)).L
    # R = cholesky_psd(Hermitian(lnew))[1]
    @debug (@assert R*R' ≈ rnew)
    Rinv = pinv(R)
    AL2 = similar(A); AR2 = similar(A)
    for x in axes(A,3)
        @views AL2[:,:,x] .= L * A[:,:,x] * Linv
        @views AR2[:,:,x] .= Rinv * A[:,:,x] * R
    end

    @debug begin    # at convergence
        @assert is_leftorth(AL2)
        @assert is_rightorth(AR2)
        qL = InfiniteUniformTensorTrain(AL2)
        qR = InfiniteUniformTensorTrain(AR2)
        q = InfiniteUniformTensorTrain(A)
        @assert marginals(qL) ≈ marginals(qR) ≈ marginals(q)
    end

    # compute C, AC
    U, c, V = svd(L * R)
    Ctilde = Diagonal(c)
    @debug (@assert U * Ctilde * V' ≈ L * R)
    ALtilde = similar(A); ARtilde = similar(A)
    for x in axes(A,3)
        @views ALtilde[:,:,x] .= U' * AL2[:,:,x] * U
        @views ARtilde[:,:,x] .= V' * AR2[:,:,x] * V
    end
    ACtilde = similar(A)
    for x in axes(ACtilde,3)
        @views ACtilde[:,:,x] .= ALtilde[:,:,x] * Ctilde
    end
    @debug begin
        ACtilde2 = similar(A)
        for x in axes(A,3)
            @views ACtilde2[:,:,x] .= Ctilde * ARtilde[:,:,x]
        end
        @assert ACtilde2 ≈ ACtilde
    end
    @debug begin
        @assert is_leftorth(ALtilde)
        @assert is_rightorth(ARtilde)
        qL = InfiniteUniformTensorTrain(ALtilde)
        qR = InfiniteUniformTensorTrain(ARtilde)
        q = InfiniteUniformTensorTrain(A)
        @assert marginals(qL) ≈ marginals(qR) ≈ marginals(q)
    end
    return ALtilde, ARtilde, ACtilde, Ctilde
end

function minAcC(AC, C; polar=false)
    ACl = reshape(permutedims(AC, (3,1,2)), :, size(AC,2))
    ACr = reshape(permutedims(AC, (1,3,2)), size(AC,1), :)
    if polar
        alg = :newton
        maxiter = 100
        UlAC = polar_left(ACl; alg, maxiter)
        UlC = polar_left(C; alg, maxiter)
        UrAC = polar_right(ACr; alg, maxiter)
        UrC = polar_right(C; alg, maxiter)    
    else
        UlAC = qrpos_left(ACl)
        UlC = qrpos_left(C)
        UrAC = qrpos_right(ACr)
        UrC = qrpos_right(C)
    end
    ALnew = permutedims(reshape(UlAC * UlC', :,size(AC,1),size(AC,2)), (2,3,1))
    ARnew = permutedims(reshape(UrC' * UrAC, size(AC,1), :, size(AC,2)), (1,3,2))
    @debug begin
        @assert is_leftorth(ALnew)
        @assert is_rightorth(ARnew)
    end
    return ALnew, ARnew
end

mutable struct VUMPSState{T}
    l  :: Matrix{T}
    r  :: Matrix{T}
    AL :: Array{T,3}
    AR :: Array{T,3}
    L  :: Matrix{T}
    R  :: Matrix{T}

    function VUMPSState(m::Integer, d::Integer, q::Integer)
        l = _initialize_posdef(m)
        r = _initialize_posdef(m)
        L = rand(d, m)
        R = rand(m, d)
        AL = rand(d, d, q)
        AR = rand(d, d, q)
        T = eltype(AR)
        return new{T}(l, r, AL, AR, L, R)
    end
end

function VUMPSState(A::Array{T,3}, d::Integer) where T
    m, _, q = size(A)
    @assert size(A,2) == m
    return VUMPSState(m, d, q)
end

dim_original(state::VUMPSState) = size(state.L, 2)
dim_trunc(state::VUMPSState) = size(state.L, 1)
dim_variables(state::VUMPSState) = size(state.AL, 3)

# resizes the fields of `state` to accomodate for a matrix mxm to be truncated to dxd
function Base.resize!(state::VUMPSState, m, d=dim_trunc(state), q=dim_variables(state))
    mold = dim_original(state)
    dold = dim_trunc(state)
    qold = dim_variables(state)
    if (m != mold || d != dold || q != qold)
        state.l = _initialize_posdef(m)
        state.r = _initialize_posdef(m)
        state.L = rand(d, m)
        state.R = rand(m, d)
        state.AL = rand(d, d, q)
        state.AR = rand(d, d, q)
    end
    return state
end

bond_dim_original(state::VUMPSState) = size(state.l, 1)
bond_dim_trunc(state::VUMPSState) = size(state.L, 1)

function one_vumps_iter!(state::VUMPSState, A;
        maxiter_ortho=10^3, maxiter_fixedpoint=10^3, tol_ortho=1e-16, tol_fixedpoint=1e-16#=,
        mix_canon = mixed_canonical_original(A; maxiter_ortho)=#)

    (; l, r, AL, AR, L, R) = state

    # bring to mixed canonical gauge
    # l += 1e-1*I; r += 1e-1*I
    # ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical!(l, r, A; maxiter_ortho)
    ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical_qr!(l, r, A; maxiter_ortho, tol_ortho)
    # ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical_original(A; maxiter_ortho)
    # ALtilde, ARtilde, ACtilde, Ctilde = mix_canon

    # fixed point of mixed transfer operators
    left_fixedpoint!(L, ALtilde, AL; maxiter=maxiter_fixedpoint, tol=tol_fixedpoint)
    right_fixedpoint!(R, ARtilde, AR; maxiter=maxiter_fixedpoint, tol=tol_fixedpoint)

    # compute AC
    AC = copy(AL)
    for x in axes(A,3)
        @views AC[:,:,x] .= L * ACtilde[:,:,x] * R
    end
    # compute C
    C = L * Ctilde * R
    # compute minAcC
    ALnew, ARnew = minAcC(AC, C)
    state.AL .= ALnew
    state.AR .= ARnew

    return ALnew, ARnew, AC, C
end

function vumps(A, d; state = VUMPSState(A, d), kwargs...)
    iterate!(state, A; kwargs...)
end

function iterate!(state::VUMPSState, A;
        maxiter_vumps=200, tol_vumps=1e-16, verbose=false,
        maxiter_ortho=10^3, maxiter_fixedpoint=10^3,
        tol_ortho=1e-16, tol_fixedpoint=1e-16,
        δs = fill(NaN, maxiter_vumps+1))

    d = bond_dim_trunc(state)
    ALC = rand(d, d, size(A, 3))
    
    δ = 1.0
    δs[1] = δ

    # mix_canon = mixed_canonical_original(A; maxiter_ortho)

    for it in 1:maxiter_vumps
        AL, AR, AC, C = one_vumps_iter!(state, A;
            maxiter_ortho, maxiter_fixedpoint, tol_ortho, tol_fixedpoint#=, mix_canon=#)
        for x in axes(A, 3)
            ALC[:,:,x] .= AL[:,:,x] * C
        end
        δ = norm(ALC - AC)
        δs[it] = δ
        δ < tol_vumps && return AL, AR
        verbose && println("iter $it. δ=$δ")
    end
    @warn "vumps not converged after $maxiter_vumps iterations. δ=$δ"
    return state.AL, state.AR
end

function one_bpvumps_iter!(state::VUMPSState, f, A, d; kw_vumps...)
    @tullio BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := 
        f(xᵢᵗ⁺¹,[xⱼᵗ,xₖᵗ,xₗᵗ],xᵢᵗ)*A[m1,n1,xₖᵗ,xᵢᵗ]*A[m2,n2,xₗᵗ,xᵢᵗ] (xⱼᵗ in 1:2, xᵢᵗ⁺¹ in 1:2)
    @cast Q[(xᵢᵗ, xⱼᵗ, m1, m2), (n1, n2, xᵢᵗ⁺¹)] := BB[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
    U, λ, V = svd(Matrix(Q))
    @cast C[m,k,xᵢᵗ,xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] (k in 1:length(λ), xᵢᵗ in 1:2, xⱼᵗ in 1:2)
    @cast Vt[m,n,xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:2
    @tullio M[m,n,xᵢᵗ,xⱼᵗ] := λ[m] * Vt[m,l,xᵢᵗ] * C[l,n,xᵢᵗ,xⱼᵗ]

    Mresh = reshape(M, size(M,1), size(M,2), :)
    p = InfiniteUniformTensorTrain(Mresh)
    # VUMPS iteration
    resize!(state, size(Mresh,1))
    Mnew, = iterate!(state, Mresh; kw_vumps...)
    Mnew ./= Mnew[1]
    q = InfiniteUniformTensorTrain(Mnew)

    Anew = reshape(Mnew, size(Mnew)[1:2]..., 2, 2) |> copy

    err = maximum(abs, only(marginals(p)) - only(marginals(q)))
    ovl = abs(dot(p, q))
    ε = abs( 1 - dot(InfiniteUniformTensorTrain(Anew), InfiniteUniformTensorTrain(A)))
    b = belief(Anew)
    return Anew, ε, err, ovl, b
end

function iterate_bp_vumps(f::Function, d::Integer;
        maxiter=50, tol=1e-3,
        showprogress = true,
        A0 = reshape(rand(2,2), 1,1,2,2),
        state = VUMPSState(size(A0,1), d, 4),
        kw_vumps...)

    errs = fill(NaN, maxiter)
    ovls = fill(NaN, maxiter)
    εs = fill(NaN, maxiter)
    beliefs = [[NaN,NaN] for _ in 1:maxiter]
    A = copy(A0)
    As = [copy(A)]
    prog = Progress(maxiter, desc="Running BP + VUMPS", dt=showprogress ? 0.1 : Inf)

    for it in 1:maxiter
        Anew, ε, err, ovl, b = one_bpvumps_iter!(state, f, A, d; kw_vumps...)
        push!(As, Anew)
        A = Anew
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
    bij = pair_belief(A, B)
    bA = sum(bij, dims=2) |> vec
    bA ./= sum(bA)
    bB = sum(bij, dims=1) |> vec
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

# function vumps_original(A, d; 
#         l = _initialize_posdef(size(A,1)),
#         r = _initialize_posdef(size(A,1)), 
#         L1 = rand(d, size(A,2)),
#         R1 = rand(size(A,1), d),
#         AL = rand(d, d, q),
#         AR = rand(d, d, q),
#         maxiter=100, tol=1e-14, verbose=true,
#         maxiter_ortho=10^3,
#         δs = fill(NaN, maxiter+1))
    

#     δ = 1.0
#     δs[1] = δ
#     ALnew = similar(AL); ARnew = similar(AR)

#     # bring to mixed canonical gauge
#     ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical(A; l, r, maxiter_ortho)

#     for it in 1:maxiter
#         # fixed point of mixed transfer operators
#         # L1new = left_fixedpoint(ALtilde, AL; L1, maxiter=5)
#         left_fixedpoint!(L, ALtilde, AL; maxiter=maxiter_fixedpoint)
#         right_fixedpoint!(R, ARtilde, AR; maxiter=5)

#         # compute AC
#         AC = zeros(d, d, q)
#         for x in axes(A,3)
#             @views AC[:,:,x] .= L1new * ACtilde[:,:,x] * R1new
#         end
#         # compute C
#         C = L1new * Ctilde * R1new
#         # compute minAcC
#         ALnew, ARnew = minAcC(AC, C)
        
#         ALC = copy(ALnew)
#         for x in axes(A, 3)
#             ALC[:,:,x] .= ALnew[:,:,x] * C
#         end
#         δ = norm(ALC - AC)
#         δs[it] = δ
#         δ < tol && return ALnew, ARnew
#         verbose && println("iter $it. δ=$δ")

#         λAL = norm(ALnew); ALnew ./= sqrt(λAL)
#         λAR = norm(ARnew); ARnew ./= sqrt(λAR)
#         L1 = L1new; R1 = R1new; AL = ALnew; AR = ARnew
#     end
#     @warn "vumps_original not converged after $maxiter iterations. δ=$δ"
#     return ALnew, ARnew
# end