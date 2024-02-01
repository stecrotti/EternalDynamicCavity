using DCAUtils
using StatsBase
using TensorTrains

max_gap_fraction = 0.2
# pc = 0.6
pf = "00595"
Y = read_fasta_alignment((@__DIR__)*"/../data/proteins/ArDCAData/data/PF$(pf)/PF$(pf)_mgap6.fasta.gz", max_gap_fraction)
Z = collect.(eachcol(Y))
unique!(Z)  # probably wrong conceptually

# theta = :auto
# W, Meff = compute_weights(Z, theta)

const q = Int(maximum(maximum.(Z)))
N = length(Z)
L = length(first(Z))

# subsample sequences
ndata = 10^2
data_subset = StatsBase.sample(1:N, ndata, replace=false)
X = Z[data_subset]

function tensortrain_from_empirical(X::Vector{Vector{T}}, q::Integer) where {T<:Integer}
    L = length(first(X))
    @assert all(length(x)==L for x in X)
    @assert all(all(<=(q), x) for x in X)

    tts = map(X) do x
        tensors = [zeros(1,1,q) for _ in 1:L]
        for l in eachindex(tensors)
            tensors[l][1,1,x[l]] = 1
        end
        TensorTrain(tensors)
    end

    return sum(tts)
end

p = tensortrain_from_empirical(X, q)
normalize!(p)

marg_emp = marginals(p)

# εs = 0.05:0.05:0.5
# using Unzip
# errs, sizes = map(εs) do ε
#     svd_trunc = TruncThresh(ε)
#     p_comp = deepcopy(p)
#     compress!(p_comp; svd_trunc)
#     marg_comp = marginals(p_comp)
#     err = maximum(maximum(abs, e-c) for (e,c) in zip(marg_emp, marg_comp))
#     sz = maximum(bond_dims(p_comp))
#     err, sz
# end |> unzip

# using Plots
# pl1 = scatter(εs, errs,
#     xlabel="Truncation error", ylabel="Error on marginals", label="")
# pl2 = scatter(εs, sizes,
#     xlabel="Truncation error", ylabel="Matrix size", label="")
# pl3 = scatter(sizes, errs,
#     xlabel="Matrix size", ylabel="Error on marginals", label="")
# pl = plot(pl1, pl2, pl3)

sizes = 10:10:120
errs = map(sizes) do sz
    svd_trunc = TruncBond(sz)
    p_comp = deepcopy(p)
    compress!(p_comp; svd_trunc)
    marg_comp = marginals(p_comp)
    err = maximum(maximum(abs, e-c) for (e,c) in zip(marg_emp, marg_comp))
end

using Plots
pl = scatter(sizes, errs,
    xlabel="Matrix size", ylabel="Max error on marginals", label="")
vline!(pl, [ndata], label="Number of sequences", ls=:dash)