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