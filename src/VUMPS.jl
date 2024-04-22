using LinearAlgebra
using TensorTrains, TensorTrains.UniformTensorTrains
using MatrixFactorizations

function mypolar(A)
    U, Σ, V = svd(A)
    Uleft = U * V'
    # Pleft = V * Diagonal(Σ) * V' |> Hermitian
    # Pright = U * Diagonal(Σ) * U' |> Hermitian
    # return Uleft, Pleft, Pright
    return Uleft
end

function polar_left(A; kw...)
    U, = MatrixFactorizations.polar(A; kw...)
    U
end
function polar_right(A; kw...)
    U, = MatrixFactorizations.polar(collect(A'); kw...)
    U'
end


d = 5
q = 2
A = rand(5, 5, q)
using JLD2
A = load("tmp.jld2")["A"]
q = size(A, 3)

# function vumps_loop(A)
# l = Matrix(1.0I, size(A,1), size(A,2))
# r = Matrix(1.0I, size(A,1), size(A,2))
# L1 = Matrix(1.0I, d, size(A,2))
# R1 = Matrix(1.0I, size(A,1), d)
l = rand(size(A,1), size(A,2))
l = l'l
r = rand(size(A,1), size(A,2))
r = r'r
L1 = rand(d, size(A,2))
R1 = rand(size(A,1), d)
AL = rand(d, d, q)
AR = rand(d, d, q)

function inner_loop(l, L1, r, R1, AL, AR)
    @views lnew = sum(A[:,:,x]' * l * A[:,:,x] for x in axes(A,3))
    λl = norm(lnew)
    lnew ./= λl
    @views rnew = sum(A[:,:,x] * r * A[:,:,x]' for x in axes(A,3))
    λr = norm(rnew)
    rnew ./= λr
    L = cholesky(Hermitian(lnew)).U
    @debug (@assert L'L ≈ lnew)
    Linv = inv(L)
    R = cholesky(Hermitian(rnew)).L
    @debug (@assert R*R' ≈ rnew)
    Rinv = inv(R)
    AL2 = similar(A); AR2 = similar(A)
    for x in axes(A,3)
        @views AL2[:,:,x] .= L * A[:,:,x] * Linv
        @views AR2[:,:,x] .= Rinv * A[:,:,x] * R
    end
    U, c, V = svd(L * R)
    Ctilde = Diagonal(c)
    @debug (@assert U * Ctilde * V' ≈ L * R)
    ALtilde = similar(A); ARtilde = similar(A)
    for x in axes(A,3)
        @views ALtilde[:,:,x] .= U' * AL2[:,:,x] * U
        @views ARtilde[:,:,x] .= V' * AR2[:,:,x] * V
    end
    ACtilde = similar(A)
    for x in axes(A,3)
        @views ACtilde[:,:,x] .= ALtilde[:,:,x] * Ctilde
    end
    @debug begin
        ACtilde2 = similar(A)
        for x in axes(A,3)
            @views ACtilde2[:,:,x] .= Ctilde * ARtilde[:,:,x]
        end
        @assert ACtilde2 ≈ ACtilde
    end
    @views L1new = sum(AL[:,:,x]' * L1 * ALtilde[:,:,x] for x in axes(A,3))
    λL1 = norm(L1new)
    L1new ./= λL1
    @views R1new = sum(ARtilde[:,:,x] * R1 * AR[:,:,x]' for x in axes(A,3))
    λR1 = norm(R1new)
    R1new ./= λR1
    # @show λL1, λR1

    AC = zeros(d, d, q)
    for x in axes(A,3)
        @views AC[:,:,x] .= L1new * ACtilde[:,:,x] * R1new
    end
    C = L1new * Ctilde * R1new

    ALnew = similar(AL); ARnew = similar(AR)
    alg = :halley
    Uleft_C = polar_left(C; alg)
    Uright_C = polar_right(C; alg)
    for x in axes(A, 3)
        Uleft_AC = polar_left(AC[:,:,x]; alg)
        Uright_AC = polar_right(AC[:,:,x]; alg)
        ALnew[:,:,x] .= Uleft_AC * Uleft_C'
        ARnew[:,:,x] .= Uright_C' * Uright_AC
    end

    # U_C = mypolar(C)
    # for x in axes(A, 3)
    #     U_AC = mypolar(AC[:,:,x])
    #     ALnew[:,:,x] = U_AC * U_C'
    #     ARnew[:,:,x] = U_C' * U_AC
    # end

    # Cinv = pinv(C; rtol = sqrt(eps(real(float(oneunit(eltype(C)))))) )
    # for x in axes(A, 3)
    #     @views ALnew[:,:,x] .= AC[:,:,x] * Cinv
    #     @views ARnew[:,:,x] .= Cinv * AC[:,:,x]
    # end
    λAL = norm(ALnew); ALnew ./= λAL
    λAR = norm(ARnew); ARnew ./= λAR
    @show norm(AL - ALnew)
    @show norm(AR - ARnew)
    @show real(marginals(InfiniteUniformTensorTrain(ALnew))[1])
    @show real(marginals(InfiniteUniformTensorTrain(ARnew))[1])

    lnew, L1new, rnew, R1new, ALnew, ARnew
end

maxiter = 300

for it in 1:maxiter
    global l, L1, r, R1, AL, AR = inner_loop(l, L1, r, R1, AL, AR)
end

qq = InfiniteUniformTensorTrain(A)
p = InfiniteUniformTensorTrain(AL)
real(marginals(qq)), real(marginals(p))

# end






# function qrpos(A::AbstractMatrix)
#     Q_, R = qr(A)
#     Q = Matrix(Q_)
#     d = @view R[diagind(R)]
#     if all(>(0), d)
#         return Q, R
#     else
#         D = Diagonal(sign.(d))
#         Q = Q * D
#         R = D * R
#         return Q, R
#     end
# end

# function left_orthonormalize(qA::InfiniteUniformTensorTrain, L0, η)
#     A = 
#     L = L0
#     L ./= norm(L)
#     Lold = L
#     AL_reshaped, L = qrpos(mul(L, A))
    
# end
