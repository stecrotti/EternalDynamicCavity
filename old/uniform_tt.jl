abstract type AbstractTransferOperator{F1<:Number,F2<:Number} end
abstract type AbstractFiniteTransferOperator{F1<:Number,F2<:Number} <: AbstractTransferOperator{F1,F2} end

struct TransferOperator{F1<:Number,F2<:Number} <: AbstractFiniteTransferOperator{F1,F2}
    A :: Array{F1,3}
    M :: Array{F2,3}
end

struct HomogeneousTransferOperator{F<:Number} <: AbstractFiniteTransferOperator{F,F}
    A :: Array{F,3}
end

function Base.convert(::Type{TransferOperator}, G::HomogeneousTransferOperator)
    TransferOperator(get_tensors(G)...)
end
TransferOperator(G::HomogeneousTransferOperator) = convert(TransferOperator, G)

function sizes(G::AbstractFiniteTransferOperator)
    A, M = get_tensors(G)
    size(A, 1), size(M, 1), size(A, 2), size(M, 2)
end

get_tensors(G::TransferOperator) = (G.A, G.M)
get_tensors(G::HomogeneousTransferOperator) = (G.A, G.A)

# the first argument `p` is the one with `M` matrices
function transfer_operator(p::AbstractUniformTensorTrain, q::AbstractUniformTensorTrain)
    return TransferOperator(_reshape1(q.tensor), _reshape1(p.tensor))
end
function transfer_operator(q::AbstractUniformTensorTrain)
    return HomogeneousTransferOperator(_reshape1(q.tensor))
end

function Base.collect(G::AbstractFiniteTransferOperator)
    A, M = get_tensors(G)
    return @tullio B[i,j,k,l] := A[i,k,x] * conj(M[j,l,x])
end

function Base.:(*)(G::AbstractFiniteTransferOperator, B::AbstractMatrix)
    A, M = get_tensors(G)
    return @tullio C[i,j] := A[i,k,x] * conj(M[j,l,x]) * B[k,l]
end

function Base.:(*)(B::AbstractMatrix, G::AbstractFiniteTransferOperator)
    A, M = get_tensors(G)
    return @tullio C[k,l] := B[i,j] * A[i,k,x] * conj(M[j,l,x])
end

function Base.:(*)(G::T, H::T) where {T<:TransferOperator}
    AG, MG = get_tensors(G)
    AH, MH = get_tensors(H)
    @tullio A[i,m,x,y] := AG[i,k,x] * AH[k,m,y]
    @tullio M[j,n,x,y] := MG[j,l,x] * MH[l,n,y] 
    return T(_reshape1(A), _reshape1(M))
end
function Base.:(*)(G::T, H::T) where {T<:HomogeneousTransferOperator}
    AG, _ = get_tensors(G)
    AH, _ = get_tensors(H)
    @tullio A[i,m,x,y] := AG[i,k,x] * AH[k,m,y]
    return T(_reshape1(A))
end

function LinearAlgebra.tr(G::AbstractFiniteTransferOperator)
    A, M = get_tensors(G)
    return sum(tr(@view A[:,:,x])*conj(tr(@view M[:,:,x])) for x in axes(A, 3))
end

function Base.:(^)(G::TransferOperator, k::Integer)
    (; L, R, Λ) = eig(G)
    d = sizes(G)
    Gk = zeros(ComplexF64, sizes(G))
    for (λ, l, r) in zip(Λ, eachrow(L), eachcol(R))
        l_ = reshape(l, d[1], d[2])
        r_ = reshape(r, d[1], d[2])
        @tullio Gk_[i,j,m,n] := λ^k * r_[i,j] * l_[m,n]
        Gk .+= Gk_
    end
    @cast Gkresh[(i,j),(k,l)] := Gk[i,k,j,l]
    U, Σ, V = svd(real(Gkresh))
    A = reshape(U * diagm(sqrt.(Σ)), d[1], d[3], :) |> collect
    M = permutedims(reshape(diagm(sqrt.(Σ)) * V', :, d[2], d[4]), (2,3,1)) |> collect
    return TransferOperator(A, M)
end

function Base.:(^)(G::HomogeneousTransferOperator, k::Integer)
    (; L, R, Λ) = eig(G)
    d = sizes(G)
    Gk = zeros(ComplexF64, sizes(G))
    for (λ, l, r) in zip(Λ, eachrow(L), eachcol(R))
        l_ = reshape(l, d[1], d[2])
        r_ = reshape(r, d[1], d[2])
        @tullio Gk_[i,j,m,n] := λ^k * r_[i,j] * l_[m,n]
        Gk .+= Gk_
    end
    @cast Gkresh[(i,j),(k,l)] := Gk[i,k,j,l]
    U, Σ, V = svd(real(Gkresh))
    A = reshape(U * diagm(sqrt.(Σ)), d[1], d[3], :) |> collect
    return HomogeneousTransferOperator(A)
end

function eig(G::AbstractTransferOperator)
    GG = collect(G)
    @cast Gresh[(i,j),(k,l)] := GG[i,j,k,l]
    E = eigen(Gresh, sortby = λ -> (-abs(λ)))
    R = eigvecs(E)
    L = pinv(R)
    Λ = eigvals(E)
    return (; L, R, Λ)
end

function leading_eig(G::AbstractTransferOperator)
    GG = collect(G)
    @cast B[(i,j),(k,l)] := GG[i,j,k,l]
    valsR, vecsR = eigsolve(B)
    valsL, vecsL = eigsolve(transpose(B))
    valsR[1] ≈ valsL[1] || @warn "Leading eigenvalue for A and Aᵀ not equal, got $(valsR[1]) and $(valsL[1])"
    λ = complex(valsL[1])
    L = vecsL[1]
    R = vecsR[1]
    d = sizes(G)
    r = reshape(R, d[1], d[2])
    l = reshape(L, d[1], d[2])
    l ./= dot(l, r)
    return (; l, r, λ)
end
function leading_eig_old(G::AbstractTransferOperator)
    L, R, Λ = eig(G)
    λ = first(Λ)
    d = sizes(G)
    r = reshape(R[:,1], d[1], d[2])
    l = reshape(L[1,:], d[1], d[2])
    return (; l, r, λ)
end

struct InfiniteTransferOperator{F<:Number,M<:AbstractMatrix{F}} <: AbstractTransferOperator{F,F}
    l :: M
    r :: M
    λ :: F
end

function Base.collect(G::InfiniteTransferOperator)
    (; l, r, λ) = G
    return @tullio B[i,j,k,m] := r[i,j] * l[k,m]
end

function sizes(G::InfiniteTransferOperator)
    (; l, r) = G
    return tuple(size(r)..., size(l)...)
end

function leading_eig(G::InfiniteTransferOperator)
    (; l, r, λ) = G
    return (; l, r, λ)
end

function eig(G::InfiniteTransferOperator)
    l, r, λ = leading_eig(G)
    L = [l]
    R = [r]
    Λ = [λ]
    return (; L, R, Λ)
end

function LinearAlgebra.tr(G::InfiniteTransferOperator)
    (; l, r, λ) = G
    return tr(l'r)
end

function infinite_transfer_operator(G::AbstractTransferOperator; lambda1::Bool=false)
    l, r, λ_ = leading_eig(G)
    λ = lambda1 ? one(λ_) : λ_
    λ = convert(eltype(r), λ)
    InfiniteTransferOperator(l, r, λ)
end

function infinite_transfer_operator(p::AbstractUniformTensorTrain, q::AbstractUniformTensorTrain)
    return infinite_transfer_operator(transfer_operator(p, q))
end

function infinite_transfer_operator(q::AbstractUniformTensorTrain)
    return infinite_transfer_operator(transfer_operator(q))
end

function LinearAlgebra.dot(p::InfiniteUniformTensorTrain, q::InfiniteUniformTensorTrain;
        G = infinite_transfer_operator(p, q),
        Ep = infinite_transfer_operator(p),
        Eq = infinite_transfer_operator(q))
    return G.λ / sqrt(abs(Ep.λ*Eq.λ))
end

# function LinearAlgebra.mul!(Y, A::InfiniteTransferOperator, B::AbstractVector)
#     Y .= vec(A.r) * dot(A.l, reshape(B, size(A.l)))
# end
function Base.:(*)(A::InfiniteTransferOperator, B::AbstractVector)
    vec(A.r) * dot(A.l, reshape(B, size(A.l)))
end

(A::InfiniteTransferOperator)(B::AbstractVector) = A * B

function gradientA!(g, p::AbstractPeriodicTensorTrain, q::AbstractPeriodicTensorTrain)
    @assert size(g) == tuple(size(q.tensor)[1:2]..., prod(size(q.tensor)[3:end]))
    L = length(p)
    M = p.tensor
    G = transfer_operator(p, q)
    A, M = get_tensors(G)
    GL = collect((G^(L-1)))
    @tullio g[a,b,x] = GL[b,j,a,l] * conj(M[l,j,x]) *($L)
end
gradientA(p, q) = gradientA!(zeros(size(q.tensor)[1:2]..., prod(size(q.tensor)[3:end])), p, q)

function gradientA!(g, q::AbstractPeriodicTensorTrain)
    L = length(q)
    A = q.tensor
    E = transfer_operator(q)
    A, M = get_tensors(E)
    EL = collect((E^(L-1)))
    @tullio g[a,b,x] = EL[b,j,a,l] * conj(A[l,j,x]) * 2 *($L)
end
gradientA(q) = gradientA!(zeros(size(q.tensor)), q)

function truncate_utt(p::UniformTensorTrain, sz::Integer;
        rng = Random.GLOBAL_RNG,
        A0 = rand(rng, sz, sz, size(p.tensor)[3:end]...),
        maxiter = 200, tol=1e-4, damp=0.0, showprogress=true)
    @assert size(A0)[1:2] == (sz, sz)
    A = _reshape1(A0)
    Anew = copy(A)
    M = p.tensor
    Mresh = _reshape1(M)
    L = p.L

    prog = Progress(maxiter, dt = showprogress ? 0.1 : Inf)
    for it in 1:maxiter
        q = UniformTensorTrain(A, L)
        normalize!(q)
        G = transfer_operator(p, q)
        E = transfer_operator(q)

        EL = E^(L-1) |> collect |> real
        GL = G^(L-1) |> collect |> real

        @tullio B[b,a,x] := GL[b,j,a,l] * Mresh[l,j,x]
        
        @cast ELresh[(b,a),(j,l)] := EL[b,j,a,l]
        @cast Bresh[(b,a), x] := B[b,a,x]
        
        for x in axes(Anew, 3)
            # k = ELresh \ Bresh[:,x]
            k = qr(ELresh, ColumnNorm()) \ Bresh[:,x]
            Anew[:,:,x] = reshape(k, Int(sqrt(length(k))), :)'
        end

        ε = norm(A - Anew) / sz
        ε < tol && return collect(_reshapeas(A, A0))
        A .= damp * A + (1-damp) * Anew 
        # next!(prog, showvalues=[("ε/tol","$ε/$tol"), ("∑ₓ(q-p)²", norm2m(q,p))])
        next!(prog, showvalues=[("ε/tol","$ε/$tol")])
    end
    return collect(_reshapeas(A, A0))
end

function truncate_utt(p::InfiniteUniformTensorTrain, sz::Integer; 
        rng = Random.GLOBAL_RNG,
        A0 = rand(rng, sz, sz, size(p.tensor)[3:end]...),
        maxiter = 200, tol=1e-4, damp=0.0, showprogress=true)
    A = _reshape1(A0)
    Anew = copy(A)
    M = p.tensor
    Mresh = _reshape1(M)

    prog = Progress(maxiter, dt = showprogress ? 0.1 : Inf)
    for it in 1:maxiter
        q = InfiniteUniformTensorTrain(A)
        G = transfer_operator(p, q)
        E = transfer_operator(q)
        Einfop = infinite_transfer_operator(E)
        Einf = collect(Einfop) |> real
        Ginfop = infinite_transfer_operator(G)
        Ginf = collect(Ginfop) |> real
        # @show Ginf
        @tullio B[b,a,x] := Ginf[b,j,a,l] * Mresh[l,j,x]
        
        @cast Einfresh[(b,a),(j,l)] := Einf[b,j,a,l]
        @cast Bresh[(b,a), x] := B[b,a,x]
        
        for x in axes(Anew, 3)
            # k = Einfresh \ Bresh[:,x]
            # k = pinv(Einfresh) * Bresh[:,x]
            k = qr(Einfresh, ColumnNorm()) \ Bresh[:,x]
            Anew[:,:,x] = reshape(k, Int(sqrt(length(k))), :)'
        end

        ε = norm(A - Anew) / sz
        ε < tol && return collect(_reshapeas(A, A0))
        A .= damp * A + (1-damp) * Anew 
        # A ./= sqrt(abs(tr(Einfop)))   # normalize A
        next!(prog, showvalues=[("ε/tol","$ε/$tol"), ("it/maxiter", "$it/$maxiter")])
    end
    return collect(_reshapeas(A, A0))
end

function truncate_variational(p::InfiniteUniformTensorTrain, sz::Integer; 
        A0 = truncate_eachtensor(p, sz).tensor,
        # A0 = rand(sz, sz, size(p.tensor)[3:end]...),
        maxiter = 200, tol=1e-12, showprogress=true)

    A = _reshape1(copy(A0))
    q = InfiniteUniformTensorTrain(A)

    Anew = deepcopy(A)
    qnew = InfiniteUniformTensorTrain(Anew)
    ds = fill(complex(NaN), maxiter + 1); ds[1] = dot(qnew, p)
    εs = fill(NaN, maxiter + 1); εs[1] = Inf 

    M = _reshape1(p.tensor)

    # prog = Progress(maxiter, dt = showprogress ? 0.1 : Inf)
    for it in 1:maxiter
        GT = infinite_transfer_operator(q, p)
        ET = infinite_transfer_operator(q)
        lG = GT.l; rG = GT.r
        lE = ET.l; rE = ET.r
        
        for x in axes(Anew, 3)
            b = vec( transpose(rG) * transpose(M[:,:,x]) * lG )
            f(a) = vec( transpose(rE) * transpose(reshape(a, size(@view A[:,:,x]))) * lE ) 
            A_, info = linsolve(f, b, vec(A[:,:,x]))
            # info.converged == 0 && @warn "$info"
            Anew[:,:,x] = reshape(A_, size(@view A[:,:,x])) |> real
        end       

        # normalize!(qnew)
        # Anew ./= maximum(abs, Anew)
        # @show Anew
        εs[it+1] = ε = norm(A - Anew) / sz 
        ds[it+1] = dot(qnew, p)
        δd = abs(ds[it+1] - ds[it])
        showprogress && println("Iter $it/$maxiter: ε=$ε/$tol, δd=$δd")
        ε < tol && return _reshapeas(A, A0) |> collect |> InfiniteUniformTensorTrain, εs, ds
        A, Anew = Anew, A
        # next!(prog, showvalues=[("ε/tol","$ε/$tol")])
    end
    @warn "Variational truncation did not converge"
    return _reshapeas(A, A0) |> collect |> InfiniteUniformTensorTrain, εs, ds
end

function truncate_utt_eigen(p::InfiniteUniformTensorTrain, sz::Integer; 
        rng = Random.GLOBAL_RNG,
        A0 = rand(rng, sz, sz, size(p.tensor)[3:end]...),
        L0 = rand(size(A0)), R0 = rand(size(A0)),
        U0 = rand(sz, size(p.tensor, 1)), V0 = rand(sz, size(p.tensor, 1)),
        maxiter = 200, tol=1e-4, damp=0.0, showprogress=true)

    A = _reshape1(A0)
    Anew = zeros(ComplexF64, size(A))
    M = _reshape1(p.tensor)

    prog = Progress(maxiter, dt = showprogress ? 0.1 : Inf)
    for _ in 1:maxiter
        q = InfiniteUniformTensorTrain(A)
        q.tensor ./= sqrt(abs(tr(infinite_transfer_operator(q)))) 
        G = transfer_operator(p, q)
        E = transfer_operator(q)
        eg = leading_eig(G)
        V = eg[:l] |> real; U = eg[:r] |> real
        ee = leading_eig(E)
        L = ee[:l] |> real; R = ee[:r] |> real
        Linv = pinv(L); Rinv = pinv(R)
        
        for x in axes(Anew, 3)
            Anew[:,:,x] = Linv * V * M[:,:,x] * U' * Rinv'
        end

        ε = norm(A - Anew) / sz
        ε < tol && return collect(_reshapeas(A, A0))
        A .= damp * A + (1-damp) * Anew 
        next!(prog, showvalues=[("ε/tol","$ε/$tol")])
    end
    return collect(_reshapeas(A, A0))
end

function truncate_eachtensor(q::T, bond_dim::Integer) where {T<:AbstractUniformTensorTrain}
    fns = Iterators.drop(fieldnames(T), 1)
    prop = (getproperty(q, fn) for fn in fns)
    A_ = _reshape1(q.tensor)
    @cast A[i, (j,x)] := A_[i,j,x]
    U, λ, V = TruncBond(bond_dim*size(A_, 3))(Matrix(A))
    Vt = reshape(V', size(V', 1), :, size(A_, 3))
    @tullio B[i,j,x] := Vt[i,k,x] * U[k,j] * λ[j]  
    return T(collect(_reshapeas(B, q.tensor)), prop...)
end

function _L(q::T) where {T<:AbstractUniformTensorTrain}
    (; l, r, λ) = infinite_transfer_operator(q)
    if all(real(e) ≤ 0 for e in eigvals(l))
        l .*= -1
    end
    # @assert l ≈ Hermitian(l) "A=$A"
    # @assert isposdef(Hermitian(l)) "A=$A"
    c = cholesky(Hermitian(l); check=false)
    return real(c.U)
end

function TensorTrains.orthogonalize_left!(q::T) where {T<:AbstractUniformTensorTrain}
    A = q.tensor
    L = _L(q)
    Linv = inv(L)
    for x in Iterators.product((1:q for q in size(A)[3:end])...)
        A[:,:,x...] .= L * A[:,:,x...] * Linv
    end
    return q
end

function _R(q::T) where {T<:AbstractUniformTensorTrain}
    (; l, r, λ) = infinite_transfer_operator(q)
    if all(real(e) ≤ 0 for e in eigvals(r))
        r .*= -1
    end
    # @assert l ≈ Hermitian(l) "A=$A"
    # @assert isposdef(Hermitian(l)) "A=$A"
    c = cholesky(Hermitian(r); check=false)
    return  real(c.L)
end

function TensorTrains.orthogonalize_right!(q::T) where {T<:AbstractUniformTensorTrain}
    A = q.tensor
    R = _R(q)
    Rinv = inv(R)
    for x in Iterators.product((1:q for q in size(A)[3:end])...)
        A[:,:,x...] .= Rinv * A[:,:,x...] * R
    end
    return q
end

function TensorTrains.dot(p::UniformTensorTrain, q::UniformTensorTrain)
    @assert p.L == q.L
    G = transfer_operator(p, q)
    G_ = collect(G)
    Gr = reshape(G_, fill(size(G.A, 1)*size(G.M, 1), 2)...)
    return tr(Gr^(p.L))
end