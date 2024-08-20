using TensorTrains: TensorTrains, SVDTrunc, summary_compact
using TensorTrains.UniformTensorTrains: InfiniteUniformTensorTrain, dot
using MatrixProductBP: MatrixProductBP, InfiniteUniformMPEM2
using ProgressMeter: ProgressUnknown, next!

struct TruncVUMPS{TI<:Integer, TF<:Real} <: SVDTrunc
    d       :: TI
    maxiter :: TI
    tol     :: TF
end
TruncVUMPS(d::Integer; maxiter=100, tol=1e-14) = TruncVUMPS(d, maxiter, tol)

TensorTrains.summary(svd_trunc::TruncVUMPS) = "VUMPS truncation to bond size m'="*string(svd_trunc.d)


function TensorTrains.compress!(A::InfiniteUniformTensorTrain; svd_trunc::TruncVUMPS=TruncVUMPS(4),
        is_orthogonal::Symbol=:none, init = rand(svd_trunc.d, prod(size(A.tensor)[3:end]), svd_trunc.d))
    (; d, maxiter, tol) = svd_trunc
    qs = size(A.tensor)[3:end]
    B = reshape(A.tensor, size(A.tensor)[1:2]..., prod(qs))
    Bperm = permutedims(B, (1,3,2))
    Btruncperm, = truncate_vumps(Bperm, d; maxiter, tol, init)
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