struct TransferOperator{F<:Number, C<:Number}
    tensor :: Array{F,4}
    L :: Matrix{C}
    R :: Matrix{C}
    Λ :: Vector{C}

    function TransferOperator(G::Array{F,4}) where {F<:Number}
        @cast Gresh[(i,j),(k,l)] := G[i,j,k,l]
        E = eigen(Gresh, sortby = λ -> (-abs(λ)))
        R = eigvecs(E)
        L = inv(R)
        Λ = eigvals(E)
        return new{F,eltype(E)}(G, L, R, Λ)
    end
end

@forward TransferOperator.tensor Base.getindex, Base.setindex!, Base.ndims, Base.axes,
    Base.reshape, Base.length, Base.iterate, Base.similar, Base.:(-)

Base.ndims(::Type{<:TransferOperator}) = 4

function transfer_operator(q::UniformTensorTrain, p::UniformTensorTrain)
    A = _reshape1(q.tensor)
    M = _reshape1(p.tensor)
    @tullio G[i,j,k,l] := A[i,k,x] * conj(M[j,l,x])
    return TransferOperator(G)
end

transfer_operator(q::UniformTensorTrain) = transfer_operator(q, q)

function Base.:(*)(G::TransferOperator, A::AbstractMatrix)
    return @tullio B[i,j] := G[i,j,k,l] * A[k,l]
end

function Base.:(*)(A::AbstractMatrix, G::TransferOperator)
    return @tullio B[k,l] := A[i,j] * G[i,j,k,l]
end

function Base.:(*)(G::TransferOperator, H::TransferOperator)
    @tullio Q[i,j,k,l] := G[i,j,m,n] * H[m,n,k,l]
    return TransferOperator(Q)
end

function slowpow(G::TransferOperator, k::Integer)
    Gk = G
    for _ in 1:k-1
        Gk = Gk * G
    end
    return Gk
end

function Base.:(^)(G::TransferOperator{F,C}, k::Integer) where {F,C}
    (; tensor, L, R, Λ) = G
    Gk = zeros(C, size(G.tensor))
    # d = Int(sqrt(length(Λ)))
    d = size(G.tensor)
    for (λ, l, r) in zip(Λ, eachrow(L), eachcol(R))
        l_ = reshape(l, d[1], d[2])
        r_ = reshape(r, d[1], d[2])
        @tullio Gk_[i,j,m,n] := λ^k * r_[i,j] * l_[m,n]
        Gk .+= Gk_
    end
    return TransferOperator(Gk)
end

function eig(G::TransferOperator)
    (; tensor, L, R, Λ) = G
    λ = first(Λ)
    # d = Int(sqrt(length(Λ)))
    d = size(G.tensor)
    r = reshape(R[:,1], d[1], d[2])
    l = reshape(L[1,:], d[1], d[2])
    (; l, r, λ)
end

Base.isapprox(G::TransferOperator, H::TransferOperator) = isapprox(G.tensor, H.tensor)


function infinite_power(G::TransferOperator; e = eig(G))
    l, r = e
    TransferOperator(@tullio Ginf_[i,j,k,m] := r[i,j] * l[k,m])
end

function LinearAlgebra.tr(G::TransferOperator)
    @tullio t := G[i,j,i,j]
end


