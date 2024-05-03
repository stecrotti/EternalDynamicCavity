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
    ε = λL1 = Inf
    L1new = copy(L1)
    for _ in 1:maxiter
        @views L1new .= sum(AL[:,:,x]' * L1 * ALtilde[:,:,x] for x in axes(ALtilde,3))
        λL1 = norm(L1new)
        L1new ./= λL1
        ε = norm(L1 - L1new)
        ε < tol && return L1new, λL1
        L1 = L1new
    end
    @info "left_fixedpoint not converged after $maxiter iterations. ε=$ε"
    return L1, λL1
end
function right_fixedpoint(ARtilde, AR; R1 = rand(size(ARtilde,1), d), maxiter=2*10^3, tol=1e-16)
    ε = λR1 = Inf
    R1new = copy(R1)
    for _ in 1:maxiter
        @views R1new .= sum(ARtilde[:,:,x] * R1 * AR[:,:,x]' for x in axes(ARtilde,3))
        λR1 = norm(R1new)
        R1new ./= λR1
        ε = norm(R1 - R1new)
        ε < tol && return R1new, λR1
        R1 = R1new
    end
    @info "right_fixedpoint not converged after $maxiter iterations. ε=$ε"
    return R1, λR1
end

function mixed_canonical(A;
        l = _initialize_posdef(A), r = _initialize_posdef(A))
    
    # compute AL, AR (the first version)
    lnew, λl = left_orthogonal(A; l, tol=1e-15)
    rnew, λr = right_orthogonal(A; r, tol=1e-15)
    @debug @assert λl ≈ λr
    λ = tr(lnew * rnew)
    lnew ./= sqrt(λ); rnew ./= sqrt(λ); A ./= sqrt(λl)
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


function vumps_original(A, d; 
        l = _initialize_posdef(A),
        r = _initialize_posdef(A), 
        L1 = rand(d, size(A,2)),
        R1 = rand(size(A,1), d),
        AL = rand(d, d, q),
        AR = rand(d, d, q),
        maxiter=100, tol=1e-14, verbose=true,
        δs = fill(NaN, maxiter+1))
    
    ALtilde, ARtilde, ACtilde, Ctilde = mixed_canonical(A; l, r)

    δ = 1.0
    δs[1] = δ
    ALnew = similar(AL); ARnew = similar(AR)

    for it in 1:maxiter
        ### fixed point of mixed transfer operators
        L1new, λL1 = left_fixedpoint(ALtilde, AL; L1, maxiter=5)
        R1new, λR1 = right_fixedpoint(ARtilde, AR; R1, maxiter=5)

        # compute AC
        AC = zeros(d, d, q)
        for x in axes(A,3)
            @views AC[:,:,x] .= L1new * ACtilde[:,:,x] * R1new
        end
        # compute C
        C = L1new * Ctilde * R1new
        # compute minAcC
        ACl = reshape(permutedims(AC, (3,1,2)), :, size(AC,2))
        ACr = reshape(permutedims(AC, (1,3,2)), size(AC,1), :)
        UlAC = polar_left(ACl)
        UlC = polar_left(C)
        UrAC = polar_right(ACr)
        UrC = polar_right(C)
        ALnew = permutedims(reshape(UlAC * UlC', :,size(AC,1),size(AC,2)), (2,3,1))
        ARnew = permutedims(reshape(UrC' * UrAC, size(AC,1), :, size(AC,2)), (1,3,2))
        @debug begin
            @assert is_leftorth(ALnew)
            @assert is_rightorth(ARnew)
        end

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



A = rand(20, 20, 4)
using JLD2
A = load("tmp.jld2")["A"]
q = size(A, 3)
m = size(A, 1)
ψ = InfiniteMPS([TensorMap(A, (ℝ^m ⊗ ℝ^q), ℝ^m)])
p = InfiniteUniformTensorTrain(A)

maxiter = 100
d = 6
δs = fill(NaN, maxiter+1)
AL, AR = vumps_original(A, d; maxiter, δs)
pL = InfiniteUniformTensorTrain(AL)
ovlL = abs(1 - dot(p, pL))
pR = InfiniteUniformTensorTrain(AR)
ovlR = abs(1 - dot(p, pR))
err_marg = max(
    maximum(abs, real(marginals(pL))[1] - real(marginals(p))[1]),
    maximum(abs, real(marginals(pR))[1] - real(marginals(p))[1])
)
@show ovlL, ovlR
B, = truncate_vumps(permutedims(A, (1,3,2)), d)
pp = InfiniteUniformTensorTrain(permutedims(B, (1,3,2)))
[real(marginals(p))[1] real(marginals(pL))[1] real(marginals(pR))[1] real(marginals(pp))[1]]

# ds = 4:8
# nsamples = 50
# maxiter = 10^3

# errs_marg, ovls = map(ds) do d
#     e, o = map(1:nsamples) do _
#         AL, AR = vumps_original(A, d; maxiter)
#         pL = InfiniteUniformTensorTrain(AL)
#         ovlL = abs(1 - dot(p, pL))
#         pR = InfiniteUniformTensorTrain(AR)
#         ovlR = abs(1 - dot(p, pR))
#         err_marg = max(
#             maximum(abs, real(marginals(pL))[1] - real(marginals(p))[1]),
#             maximum(abs, real(marginals(pR))[1] - real(marginals(p))[1])
#         )
#         ovl = max(abs(1 - dot(pL, p)), abs(1 - dot(pR, p))) 
#         err_marg, abs(1 - ovl)
#     end |> unzip
#     mean(log, e), mean(log, o)
# end |> unzip

# using Plots
# pl_marg = plot(ds, errs_marg, label="error on marginals", m=:o, xlabel="bond dim")
# pl_ovl = plot(ds, ovls, label="1 - ovl", m=:o, xlabel="bond dim")
# plot(pl_marg, pl_ovl, legend=:bottomright, layout=(2,1), size=(400,600))


# for it in 1:maxiter
#     global l, L1, r, R1, AL, AR = inner_loop(l, L1, r, R1, AL, AR)
# end


# function inner_loop(l, L1, r, R1, AL, AR)
#     # @views lnew = sum(A[:,:,x]' * l * A[:,:,x] for x in axes(A,3))
#     # λl = norm(lnew)
#     # lnew ./= λl
#     lnew, λl = left_orthogonal(A; l, maxiter=1)
#     # @views rnew = sum(A[:,:,x] * r * A[:,:,x]' for x in axes(A,3))
#     # λr = norm(rnew)
#     # rnew ./= λr
#     rnew, λl = right_orthogonal(A; r, maxiter=1)
#     @debug @assert λl ≈ λr
#     λ = tr(lnew * rnew)
#     lnew ./= sqrt(λ); rnew ./= sqrt(λ)
#     L = cholesky(Hermitian(lnew)).U
#     @debug (@assert L'L ≈ lnew)
#     Linv = inv(L)
#     R = cholesky(Hermitian(rnew)).L
#     @debug (@assert R*R' ≈ rnew)
#     Rinv = inv(R)
#     AL2 = similar(A); AR2 = similar(A)
#     for x in axes(A,3)
#         @views AL2[:,:,x] .= L * A[:,:,x] * Linv
#         @views AR2[:,:,x] .= Rinv * A[:,:,x] * R
#     end
#     # at convergence, AL and AR give the identity when contracted on 2 indices
#     @debug begin
#         @tullio x[j,k] := AL2[i,j,x] * AL2[i,k,x]
#         @tullio y[j,k] := AR2[j,l,x] * AR2[k,l,x]
#     end
#     U, c, V = svd(L * R)
#     Ctilde = Diagonal(c)
#     @debug (@assert U * Ctilde * V' ≈ L * R)
#     ALtilde = similar(A); ARtilde = similar(A)
#     for x in axes(A,3)
#         @views ALtilde[:,:,x] .= U' * AL2[:,:,x] * U
#         @views ARtilde[:,:,x] .= V' * AR2[:,:,x] * V
#     end

#     ACtilde = similar(A)
#     for x in axes(A,3)
#         @views ACtilde[:,:,x] .= ALtilde[:,:,x] * Ctilde
#     end
#     @debug begin
#         ACtilde2 = similar(A)
#         for x in axes(A,3)
#             @views ACtilde2[:,:,x] .= Ctilde * ARtilde[:,:,x]
#         end
#         @assert ACtilde2 ≈ ACtilde
#     end

#     @views L1new = sum(AL[:,:,x]' * L1 * ALtilde[:,:,x] for x in axes(A,3))
#     λL1 = norm(L1new)
#     L1new ./= λL1
#     @views R1new = sum(ARtilde[:,:,x] * R1 * AR[:,:,x]' for x in axes(A,3))
#     λR1 = norm(R1new)
#     R1new ./= λR1
#     # @show λL1, λR1

#     AC = zeros(d, d, q)
#     for x in axes(A,3)
#         @views AC[:,:,x] .= L1new * ACtilde[:,:,x] * R1new
#     end
#     C = L1new * Ctilde * R1new

#     ALnew = similar(AL); ARnew = similar(AR)
#     alg = :halley
#     Uleft_C = polar_left(C; alg)
#     Uright_C = polar_right(C; alg)
#     for x in axes(A, 3)
#         Uleft_AC = polar_left(AC[:,:,x]; alg)
#         Uright_AC = polar_right(AC[:,:,x]; alg)
#         ALnew[:,:,x] .= Uleft_AC * Uleft_C'
#         ARnew[:,:,x] .= Uright_C' * Uright_AC
#     end

#     # U_C = mypolar(C)
#     # for x in axes(A, 3)
#     #     U_AC = mypolar(AC[:,:,x])
#     #     ALnew[:,:,x] = U_AC * U_C'
#     #     ARnew[:,:,x] = U_C' * U_AC
#     # end

#     # Cinv = pinv(C; rtol = sqrt(eps(real(float(oneunit(eltype(C)))))) )
#     # for x in axes(A, 3)
#     #     @views ALnew[:,:,x] .= AC[:,:,x] * Cinv
#     #     @views ARnew[:,:,x] .= Cinv * AC[:,:,x]
#     # end
#     λAL = norm(ALnew); ALnew ./= λAL
#     λAR = norm(ARnew); ARnew ./= λAR
#     @show norm(AL - ALnew)
#     # @show norm(AR - ARnew)
#     # @show real(marginals(InfiniteUniformTensorTrain(ALnew))[1])
#     # @show real(marginals(InfiniteUniformTensorTrain(ARnew))[1])

#     lnew, L1new, rnew, R1new, ALnew, ARnew
# end

# maxiter = 200

# for it in 1:maxiter
#     global l, L1, r, R1, AL, AR = inner_loop(l, L1, r, R1, AL, AR)
# end

# qq = InfiniteUniformTensorTrain(A)
# p = InfiniteUniformTensorTrain(AL)
# real(marginals(qq)), real(marginals(p))

# end








# function left_orthonormalize(qA::InfiniteUniformTensorTrain, L0, η)
#     A = 
#     L = L0
#     L ./= norm(L)
#     Lold = L
#     AL_reshaped, L = qrpos(mul(L, A))
    
# end
