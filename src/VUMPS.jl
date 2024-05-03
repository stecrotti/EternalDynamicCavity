using LinearAlgebra
using TensorTrains, TensorTrains.UniformTensorTrains
using MPSExperiments
using MatrixFactorizations
using TensorKit, MPSKit
using Tullio
using Random
using ProgressMeter
using Unzip
using Statistics

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

function _initialize_posdef(A; rng=Random.default_rng())
    l = rand(rng, size(A,1), size(A,2))
    l = l'l
    l ./= norm(l)
    return l
end

function left_orthogonal(A; l = _initialize_posdef(A), 
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

function right_orthogonal(A; r = _initialize_posdef(A), 
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

function is_leftorth(AL)
    @tullio x[j,k] := AL[i,j,x] * AL[i,k,x]
    x ≈ I
end
function is_rightorth(AR)
    @tullio y[j,k] := AR[j,l,x] * AR[k,l,x]
    y ≈ I
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

function mixed_canonical(A; l = _initialize_posdef(A), r = _initialize_posdef(A),
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
    L = cholesky(Hermitian(lnew)).U
    Linv = inv(L)
    R = cholesky(Hermitian(rnew)).L
    @debug (@assert R*R' ≈ rnew)
    Rinv = inv(R)
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

function vumps_original(A, d; 
        l = _initialize_posdef(A),
        r = _initialize_posdef(A), 
        L1 = rand(d, size(A,2)),
        R1 = rand(size(A,1), d),
        AL = rand(d, d, q),
        AR = rand(d, d, q),
        maxiter=100, tol=1e-14, verbose=true,
        maxiter_ortho=10^3,
        δs = fill(NaN, maxiter+1))
    

    δ = 1.0
    δs[1] = δ
    ALnew = similar(AL); ARnew = similar(AR)

    # bring to mixed canonical gauge
    ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical(A; l, r, maxiter_ortho)

    for it in 1:maxiter
        # fixed point of mixed transfer operators
        # L1new = left_fixedpoint(ALtilde, AL; L1, maxiter=5)
        left_fixedpoint!(L, ALtilde, AL; maxiter=maxiter_fixedpoint)
        right_fixedpoint!(R, ARtilde, AR; maxiter=5)

        # compute AC
        AC = zeros(d, d, q)
        for x in axes(A,3)
            @views AC[:,:,x] .= L1new * ACtilde[:,:,x] * R1new
        end
        # compute C
        C = L1new * Ctilde * R1new
        # compute minAcC
        ALnew, ARnew = minAcC(AC, C)
        
        ALC = copy(ALnew)
        for x in axes(A, 3)
            ALC[:,:,x] .= ALnew[:,:,x] * C
        end
        δ = norm(ALC - AC)
        δs[it] = δ
        δ < tol && return ALnew, ARnew
        verbose && println("iter $it. δ=$δ")

        λAL = norm(ALnew); ALnew ./= sqrt(λAL)
        λAR = norm(ARnew); ARnew ./= sqrt(λAR)
        L1 = L1new; R1 = R1new; AL = ALnew; AR = ARnew
    end
    @warn "vumps_original not converged after $maxiter iterations. δ=$δ"
    return ALnew, ARnew
end

function one_vumps_iter!(A, l, r, AL, AR, L, R;
        maxiter_ortho=10, maxiter_fixedpoint=10)
    # bring to mixed canonical gauge
    ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical!(l, r, A; maxiter_ortho)

    # fixed point of mixed transfer operators
    left_fixedpoint!(L, ALtilde, AL; maxiter=maxiter_fixedpoint)
    right_fixedpoint!(R, ARtilde, AR; maxiter=maxiter_fixedpoint)

    # compute AC
    AC = copy(AL)
    for x in axes(A,3)
        @views AC[:,:,x] .= L * ACtilde[:,:,x] * R
    end
    # compute C
    C = L * Ctilde * R
    # compute minAcC
    AL, AR = minAcC(AC, C)

    return AL, AR, AC, C
end

function vumps(A, d;
        maxiter=200, tol=1e-14, verbose=false,
        maxiter_ortho=10, maxiter_fixedpoint=10,
        δs = fill(NaN, maxiter+1))

    l = _initialize_posdef(A); r = _initialize_posdef(A)
    L = rand(d, size(A,2)); R = rand(size(A,1), d)
    AL = rand(d, d, q); AR = rand(d, d, q)
    AC = zeros(d, d, q); ALC = rand(d, d, q)
    
    δ = 1.0
    δs[1] = δ
    for it in 1:maxiter
        AL, AR, AC, C = one_vumps_iter!(A, l, r, AL, AR, L, R;
            maxiter_ortho, maxiter_fixedpoint)
        
        for x in axes(A, 3)
            ALC[:,:,x] .= AL[:,:,x] * C
        end
        δ = norm(ALC - AC)
        δs[it] = δ
        δ < tol && return AL, AR
        verbose && println("iter $it. δ=$δ")
    end
    @warn "vumps not converged after $maxiter iterations. δ=$δ"
    return AL, AR
end

using Logging
Logging.disable_logging(Logging.Info)

# A = rand(50, 50, 4)
using JLD2
A = load("tmp.jld2")["A"]
q = size(A, 3)
m = size(A, 1)
ψ = InfiniteMPS([TensorMap(A, (ℝ^m ⊗ ℝ^q), ℝ^m)])
p = InfiniteUniformTensorTrain(A)

maxiter = 200
maxiter_ortho = 2
maxiter_fixedpoint = 2
tol = 1e-6

d = 6
δs = fill(NaN, maxiter+1)
# AL, AR = vumps_original(A, d; maxiter, δs)
AL, AR = vumps(A, d; maxiter, maxiter_ortho, maxiter_fixedpoint, tol, δs)

pL = InfiniteUniformTensorTrain(AL)
ovlL = abs(1 - dot(p, pL))
pR = InfiniteUniformTensorTrain(AR)
ovlR = abs(1 - dot(p, pR))
err_marg = max(
    maximum(abs, real(marginals(pL))[1] - real(marginals(p))[1]),
    maximum(abs, real(marginals(pR))[1] - real(marginals(p))[1])
)
@show ovlL, ovlR
Aperm = permutedims(A, (1,3,2))
B, = truncate_vumps(Aperm, d)
pp = InfiniteUniformTensorTrain(permutedims(B, (1,3,2)))
[real(marginals(p))[1] real(marginals(pL))[1] real(marginals(pR))[1] real(marginals(pp))[1]]

# ds = 4:8
# nsamples = 50
# maxiter = 10^2

# errs_marg, ovls = map(ds) do d
#     e, o = map(1:nsamples) do _
#         AL, AR = vumps(A, d; maxiter)
#         pL = InfiniteUniformTensorTrain(AL)
#         ovlL = abs(1 - dot(p, pL))
#         pR = InfiniteUniformTensorTrain(AR)
#         ovlR = abs(1 - dot(p, pR))
#         err_marg = max(
#             maximum(abs, real(marginals(pL))[1] - real(marginals(p))[1]),
#             maximum(abs, real(marginals(pR))[1] - real(marginals(p))[1])
#         )
#         ovl = max(abs(1 - dot(pL, p)), abs(1 - dot(pR, p))) 
#         err_marg, ovl
#     end |> unzip
#     mean(e), mean(o)
# end |> unzip

# using Plots
# pl_marg = plot(ds, errs_marg, label="error on marginals", m=:o, xlabel="bond dim")
# pl_ovl = plot(ds, ovls, label="1 - ovl", m=:o, xlabel="bond dim")
# plot(pl_marg, pl_ovl, legend=:bottomleft, layout=(2,1), size=(400,600), margin=10Plots.mm, yaxis=:log10)
