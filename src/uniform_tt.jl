abstract type AbstractTransferOperator{F<:Number} end
abstract type AbstractFiniteTransferOperator{F<:Number} <: AbstractTransferOperator{F} end

struct TransferOperator{F<:Number} <: AbstractFiniteTransferOperator{F}
    A :: Array{F,3}
    M :: Array{F,3}
end

struct HomogeneousTransferOperator{F<:Number} <: AbstractFiniteTransferOperator{F}
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
    L, R, Λ = eig(G)
    λ = first(Λ)
    d = sizes(G)
    r = reshape(R[:,1], d[1], d[2])
    l = reshape(L[1,:], d[1], d[2])
    return (; l, r, λ)
end

struct InfiniteTransferOperator{F<:Number,M<:AbstractMatrix{F}} <: AbstractTransferOperator{F}
    l :: M
    r :: M
    λ :: F
end

function Base.collect(G::InfiniteTransferOperator)
    (; l, r, λ) = G
    return @tullio B[i,j,k,m] := r[i,j] * l[k,m] * λ
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
    return λ * tr(l'r)
end

function infinite_transfer_operator(G::AbstractTransferOperator)
    l, r, λ = leading_eig(G)
    InfiniteTransferOperator(l, r, λ)
end

function infinite_transfer_operator(p::AbstractUniformTensorTrain, q::AbstractUniformTensorTrain)
    return infinite_transfer_operator(transfer_operator(p, q))
end

function infinite_transfer_operator(q::AbstractUniformTensorTrain)
    return infinite_transfer_operator(transfer_operator(q))
end

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
        # q.tensor ./= norm(q)
        G = transfer_operator(p, q)
        E = transfer_operator(q)

        EL = E^(L-1) |> collect |> real
        GL = G^(L-1) |> collect |> real
        # @show GL

        @tullio B[b,a,x] := GL[b,j,a,l] * Mresh[l,j,x]
        
        @cast ELresh[(b,a),(j,l)] := EL[b,j,a,l]
        @cast Bresh[(b,a), x] := B[b,a,x]
        
        for x in axes(Anew, 3)
            k = ELresh \ Bresh[:,x]
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
    for _ in 1:maxiter
        q = InfiniteUniformTensorTrain(A)
        q.tensor ./= sqrt(abs(tr(infinite_transfer_operator(q)))) 
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
            k = Einfresh \ Bresh[:,x]
            Anew[:,:,x] = reshape(k, Int(sqrt(length(k))), :)'
        end

        ε = norm(A - Anew) / sz
        ε < tol && return collect(_reshapeas(A, A0))
        A .= damp * A + (1-damp) * Anew 
        # A ./= sqrt(abs(tr(Einfop)))   # normalize A
        next!(prog, showvalues=[("ε/tol","$ε/$tol")])
    end
    return collect(_reshapeas(A, A0))
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