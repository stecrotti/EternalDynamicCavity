# struct TransferOperator{F<:Number, C<:Number}
#     tensor :: Array{F,4}
#     L :: Matrix{C}
#     R :: Matrix{C}
#     Λ :: Vector{C}

#     function TransferOperator(G::Array{F,4}) where {F<:Number}
#         @cast Gresh[(i,j),(k,l)] := G[i,j,k,l]
#         E = eigen(Gresh, sortby = λ -> (-abs(λ)))
#         R = eigvecs(E)
#         L = pinv(R)
#         Λ = eigvals(E)
#         return new{F,eltype(E)}(G, L, R, Λ)
#     end
# end

# @forward TransferOperator.tensor Base.getindex, Base.setindex!, Base.ndims, Base.axes,
#     Base.reshape, Base.length, Base.iterate, Base.similar, Base.:(-)

# Base.ndims(::Type{<:TransferOperator}) = 4

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

function transfer_operator(q::AbstractUniformTensorTrain, p::AbstractUniformTensorTrain)
    # A = _reshape1(q.tensor)
    # M = _reshape1(p.tensor)
    # @tullio G[i,j,k,l] := A[i,k,x] * conj(M[j,l,x])
    return TransferOperator(_reshape1(q.tensor), _reshape1(p.tensor))
end
function transfer_operator(q::AbstractUniformTensorTrain)
    return HomogeneousTransferOperator(_reshape1(q.tensor))
end

function Base.collect(G::AbstractFiniteTransferOperator)
    A, M = get_tensors(G)
    return @tullio B[i,j,k,l] := A[i,k,x] * conj(M[j,l,x])
end

# transfer_operator(q::AbstractUniformTensorTrain) = transfer_operator(q, q)

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
    # @tullio t := G[i,j,i,j]
    A, M = get_tensors(G)
    return sum(tr(@view A[:,:,x])*conj(tr(@view M[:,:,x])) for x in axes(A, 3))
end

function Base.:(^)(G::AbstractTransferOperator, k::Integer)
    Gk = G
    for _ in 1:k-1
        Gk = Gk * G
    end
    return Gk
end

# function Base.:(^)(G::AbstractTransferOperator, k::Integer)
#     (; L, R, Λ) = eig(G)
#     d = sizes(G)
#     for (λ, l, r) in zip(Λ, eachrow(L), eachcol(R))
#         l_ = reshape(l, d[1], d[2])
#         r_ = reshape(r, d[1], d[2])
#         @tullio Gk_[i,j,m,n] := λ^k * r_[i,j] * l_[m,n]
#         Gk .+= Gk_
#     end
#     return TransferOperator(Gk)
# end

# function eig(G::TransferOperator)
#     (; tensor, L, R, Λ) = G
#     λ = first(Λ)
#     # d = Int(sqrt(length(Λ)))
#     d = size(G.tensor)
#     r = reshape(R[:,1], d[1], d[2])
#     l = reshape(L[1,:], d[1], d[2])
#     (; l, r, λ)
# end
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

# Base.isapprox(G::TransferOperator, H::TransferOperator) = isapprox(G.tensor, H.tensor)

function infinite_transfer_operator(G::AbstractTransferOperator)
    l, r, λ = leading_eig(G)
    InfiniteTransferOperator(l, r, λ)
end

# function infinite_power(G::TransferOperator; e = leading_eig(G))
#     l, r = e
#     TransferOperator(@tullio Ginf_[i,j,k,m] := r[i,j] * l[k,m])
# end



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
        G = transfer_operator(q, p)
        E = transfer_operator(q)

        # EL = prod(fill(E, L-1)).tensor
        # GL = prod(fill(G, L-1)).tensor
        EL = E^(L-1) |> collect .|> real
        GL = G^(L-1) |> collect .|> real
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

        G = transfer_operator(q, p)
        E = transfer_operator(q)
        Einfop = infinite_power(E)
        Einf = Einfop.tensor .|> real
        Ginfop = infinite_power(G)
        Ginf = Ginfop.tensor .|> real
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
        A ./= sqrt(abs(tr(Einfop)))   # normalize A
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

        G = transfer_operator(q, p)
        E = transfer_operator(q)
        eg = eig(G)
        V = eg[:l] |> real; U = eg[:r] |> real
        ee = eig(E)
        L = ee[:l] |> real; R = ee[:r] |> real
        Linv = pinv(L); Rinv = pinv(R)
        
        for x in axes(Anew, 3)
            Anew[:,:,x] = Linv * V * M[:,:,x] * U' * Rinv'
        end

        ε = norm(A - Anew) / sz
        ε < tol && return collect(_reshapeas(A, A0))
        A .= damp * A + (1-damp) * Anew 
        normalize!(q)
        next!(prog, showvalues=[("ε/tol","$ε/$tol")])
    end
    return collect(_reshapeas(A, A0))
end