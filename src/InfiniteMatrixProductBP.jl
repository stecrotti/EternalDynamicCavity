__precompile__(false)

module InfiniteMatrixProductBP

using TensorTrains: TensorTrains, SVDTrunc, summary_compact, _reshape1
using TensorTrains.UniformTensorTrains: InfiniteUniformTensorTrain, dot, rand_infinite_uniform_tt
using MatrixProductBP: MatrixProductBP, InfiniteUniformMPEM2, means
using ProgressMeter: ProgressUnknown, next!

using TensorKit: TensorMap, ⊗, ℝ, id, storagetype
using MPSKit: InfiniteMPS, DenseMPO, VUMPS, approximate, dot, add_util_leg, site_type, physicalspace

export TruncVUMPS, CB_BPVUMPS

struct TruncVUMPS{TI<:Integer, TF<:Real} <: SVDTrunc
    d       :: TI
    maxiter :: TI
    tol     :: TF
end
TruncVUMPS(d::Integer; maxiter=100, tol=1e-14) = TruncVUMPS(d, maxiter, tol)

TensorTrains.summary(svd_trunc::TruncVUMPS) = "VUMPS truncation to bond size m'="*string(svd_trunc.d)

function truncate_vumps(A::Array{F,3}, d; 
        init = rand(d, size(A,2), d),
        maxiter = 100, kw_vumps...) where {F}
    ψ = InfiniteMPS([TensorMap(init, (ℝ^d ⊗ ℝ^size(A,2)), ℝ^d)])
    Q = size(A, 2)
    m = size(A, 1)
    @assert size(A, 3) == m
    t = TensorMap(A,(ℝ^m ⊗ ℝ^Q), ℝ^m) # the same but as a type digestible by MPSKit.jl
    ψ₀ = InfiniteMPS([t])
    II = DenseMPO([add_util_leg(id(storagetype(site_type(ψ₀)), physicalspace(ψ₀, i)))
        for i in 1:length(ψ₀)])
    alg = VUMPS(; maxiter, verbosity=0, kw_vumps...) # variational approximation algorithm
    # alg = IDMRG1(; maxiter)
    @assert typeof(ψ) == typeof(ψ₀)
    ψ_, = approximate(ψ, (II, ψ₀), alg)   # do the truncation
    @assert typeof(ψ) == typeof(ψ_)

    ovl = abs(dot(ψ_, ψ₀))
    B = reshape(only(ψ_.AL).data, d, Q, d)
    return B, ovl, ψ_
end

function TensorTrains.compress!(A::InfiniteUniformTensorTrain; svd_trunc::TruncVUMPS=TruncVUMPS(4),
        is_orthogonal::Symbol=:none, init = rand_infinite_uniform_tt(svd_trunc.d, size(A.tensor)[3:end]...))
    (; d, maxiter, tol) = svd_trunc
    qs = size(A.tensor)[3:end]
    B = reshape(A.tensor, size(A.tensor)[1:2]..., prod(qs))
    Bperm = permutedims(B, (1,3,2))
    # reduce or expand `init` to match bond dimension `svd_trunc.d`
    s = size(init.tensor)
    init_resized = if s[1] != svd_trunc.d
        init_ = InfiniteUniformTensorTrain(zeros(svd_trunc.d, svd_trunc.d, size(A.tensor)[3:end]...))
        init_.tensor[1:s[1],1:s[2],fill(:,length(qs))...] = init.tensor
        init_
    else
        init
    end
    @debug begin
        if size(permutedims(_reshape1(init_resized.tensor), (1,3,2))) != size(rand(svd_trunc.d, prod(size(A.tensor)[3:end]), svd_trunc.d))
            @show size(permutedims(_reshape1(init_resized.tensor), (1,3,2))) size(rand(svd_trunc.d, prod(size(A.tensor)[3:end]), svd_trunc.d))
        end
    end
    Btruncperm, = truncate_vumps(Bperm, d; maxiter, tol, init = permutedims(_reshape1(init_resized.tensor), (1,3,2)))
    Btrunc = permutedims(Btruncperm, (1,3,2))
    A.tensor = reshape(Btrunc, size(Btrunc)[1:2]..., qs...)
    return A
end


MatrixProductBP.default_truncator(::Type{<:InfiniteUniformMPEM2}) = TruncVUMPS(4)
    
struct CB_BPVUMPS{TP<:ProgressUnknown, F, M2<:InfiniteUniformMPEM2}
    prog :: TP
    m    :: Vector{Vector{Vector{Float64}}} 
    Δs   :: Vector{Float64}     # convergence error on marginals
    A    :: Vector{Vector{M2}}     
    εs   :: Vector{Float64}     # convergence error on messages
    f    :: F

    function CB_BPVUMPS(bp::MatrixProductBP.MPBPStationary{G, T, V, M2}; showprogress::Bool=true, f::F=(x,i)->x, info="") where {G, T, V, M2, F}
        dt = showprogress ? 0.1 : Inf
        isempty(info) || (info *= "\n")
        prog = ProgressUnknown(desc=info*"Running MPBP: iter", dt=dt, showspeed=true)
        TP = typeof(prog)
        m = [means(f, bp)]
        Δs = zeros(0)
        A = [deepcopy(bp.μ.v)]
        εs = zeros(0)
        new{TP,F,M2}(prog, m, Δs, A, εs, f)
    end
end

function (cb::CB_BPVUMPS)(bp::MatrixProductBP.MPBPStationary, it::Integer, svd_trunc::SVDTrunc)
    marg_new = means(cb.f, bp)
    marg_old = cb.m[end]
    Δ = isempty(marg_new) ? NaN : maximum(maximum(abs, mn .- mo) for (mn, mo) in zip(marg_new, marg_old))
    push!(cb.Δs, Δ)
    push!(cb.m, marg_new)
    A_new = bp.μ
    A_old = cb.A[end]
    ε = isempty(A_new) ? NaN : maximum(abs, 1 - dot(Anew, Aold) for (Anew, Aold) in zip(A_new, A_old))
    push!(cb.εs, ε)
    push!(cb.A, deepcopy(bp.μ))
    next!(cb.prog, showvalues=[(:Δ,Δ), (:trunc, summary_compact(svd_trunc))])
    flush(stdout)
    return Δ
end

end
