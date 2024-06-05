using TensorTrains: TensorTrains, SVDTrunc
using TensorTrains.UniformTensorTrains: InfiniteUniformTensorTrain
using MatrixProductBP: MatrixProductBP, InfiniteUniformMPEM2

struct TruncVUMPS{TI<:Integer, TF<:Real} <: SVDTrunc
    d       :: TI
    maxiter :: TI
    tol     :: TF
end
TruncVUMPS(d::Integer; maxiter=100, tol=1e-14) = TruncVUMPS(d, maxiter, tol)

TensorTrains.summary(svd_trunc::TruncVUMPS) = "VUMPS truncation to bond size m'="*string(svd_trunc.d)


function TensorTrains.compress!(A::InfiniteUniformTensorTrain; svd_trunc::TruncVUMPS=TruncVUMPS(4),
        is_orthogonal::Symbol=:none)
    (; d, maxiter, tol) = svd_trunc
    qs = size(A.tensor)[3:end]
    B = reshape(A.tensor, size(A.tensor)[1:2]..., prod(qs))
    Bperm = permutedims(B, (1,3,2))
    Btruncperm, = truncate_vumps(Bperm, d; maxiter, tol)
    Btrunc = permutedims(Btruncperm, (1,3,2))
    A.tensor = reshape(Btrunc, size(Btrunc)[1:2]..., qs...)
    return A
end


MatrixProductBP.default_truncator(::Type{<:InfiniteUniformMPEM2}) = TruncVUMPS(4)
    