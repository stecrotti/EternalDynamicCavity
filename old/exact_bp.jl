using MatrixProductBP, MatrixProductBP.Models
using Revise
using InvertedIndices, ProgressMeter, LinearAlgebra

struct ExactMessage{F<:Real,N} 
    L  :: Int
    qs :: Vector{Int}
    p  :: Array{F,N}

    function ExactMessage(L::Int, qs::Vector{Int}, p::Array{F,N}) where {F,N}
        @assert size(p) == tuple(repeat(collect(qs), L)...)
        new{F,N}(L, qs, p)
    end
end

Base.length(msg::ExactMessage) = msg.L

function rand_exact_message(L::Integer, qs)
    p = rand(repeat(collect(qs), L)...)
    p ./= sum(p)
    return ExactMessage(L, collect(qs), p)
end
function uniform_exact_message(L::Integer, qs)
    p = ones(repeat(collect(qs), L)...)
    p ./= sum(p)
    return ExactMessage(L, collect(qs), p)
end
function zero_exact_message(L::Integer, qs)
    p = zeros(repeat(collect(qs), L)...)
    return ExactMessage(L, collect(qs), p)
end

function (msg::ExactMessage)(x)
    msg.p[reduce(vcat, x)...]
end

function time_marginals(msg::ExactMessage)
    (; L, qs, p) = msg
    map(1:L) do i
        ids = ((i-1)*length(qs)+1):(i*length(qs))
        s = sum(p, dims=((1:ndims(p))[Not(ids)]))
        dropdims(s, dims=tuple((1:ndims(p))[Not(ids)]...))
    end
end

function f_bp(msgs_in::Vector{M2}, wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, 
        qi::Integer, j_index::Integer, periodic=false; 
        showprogress=false) where {U<:BPFactor,M2<:ExactMessage,F}
    
    T = length(msgs_in[1]) - 1
    @assert all(length(a) == T + 1 for a in msgs_in)
    @assert length(wᵢ) == T + 1
    @assert j_index in eachindex(msgs_in)
    qs = msgs_in[1].qs
    qj = qs[j_index]
    msg_out = zero_exact_message(T+1, [qi, qj])
    
    dt = showprogress ? 1.0 : Inf
    prog = Progress(length(msg_out), dt=dt, desc="Computing outgoing message")
    for xᵢ in Iterators.product(fill(1:qi, T+1)...)
        for xₙ in Iterators.product((keys(min.p) for min in msgs_in)...)
            w = 1.0
            for t in (periodic ? (1:T+1) : (1:T))
                w *= wᵢ[t](xᵢ[mod1(t+1, T+1)], [xₖ[t] for xₖ in xₙ], xᵢ[t])
            end
            for k in eachindex(xₙ)
                k == j_index && continue
                w *= msgs_in[k].p[reduce(vcat, [xₙ[k][t], xᵢ[t]] for t in 1:T+1)...]
            end
            w *= prod(ϕᵢ[t][xi] for (t,xi) in pairs(xᵢ))
            msg_out.p[reduce(vcat, [xᵢ[t], xₙ[j_index][t]] for t in 1:T+1)...] += w
        end
        next!(prog)
    end
    msg_out.p ./= sum(msg_out.p)
    return msg_out
end

function pair_belief(msg::ExactMessage)
    pair_belief(msg, msg2)
end

function pair_belief(msg1::ExactMessage, msg2::ExactMessage)
    msg2_ = permutedims(msg2.p, tuple((isodd(i) ? i+1 : i-1 for i in 1:ndims(msg2.p))...))
    pb = msg1.p .* msg2_
    pb ./= sum(pb)
    ExactMessage(msg1.L, msg1.qs, pb)
end

function beliefs_from_pair(pb::ExactMessage)
    s = sum(pb.p, dims=(2:2:ndims(pb.p)))
    b = dropdims(s, dims=tuple((2:2:ndims(pb.p))...))
    b1 = ExactMessage(pb.L, [pb.qs[1]], b)
    s = sum(pb.p, dims=(1:2:ndims(pb.p)))
    b = dropdims(s, dims=tuple((1:2:ndims(pb.p))...))
    b2 = ExactMessage(pb.L, [pb.qs[2]], b)
    b1, b2
end

function beliefs(msg::ExactMessage)
    pb = pair_belief(msg)
    beliefs_from_pair(pb)
end

function iterate_bp(T, wᵢ, ϕᵢ, z, periodic::Bool; maxiter=10, tol=1e-5, damp=0.0,
        showprogress = false)
    msg = rand_exact_message(T+1, (2,2))
    prog = Progress(maxiter, dt = showprogress ? 0.1 : Inf)
    εs = fill(NaN, maxiter)
    for it in 1:maxiter
        print("BP iter $it of $maxiter: ")
        msg_new = f_bp(fill(msg, z), wᵢ, ϕᵢ, 2, 1, periodic; showprogress=false)
        ε = norm(msg.p - msg_new.p) / length(msg.p)
        ε < tol && return msg, it, εs
        next!(prog, showvalues=[(:it, it), (:ε, ε)])
        println("ε = ", ε)
        εs[it] = ε
        msg.p .= damp * msg.p + (1-damp) * msg_new.p
    end
    return msg, maxiter, εs
end

function bp_star(T, wᵢ, ϕᵢ, z, periodic::Bool; showprogress = false)
    msg_in = f_bp([rand_exact_message(T+1, (2,2))], wᵢ, ϕᵢ, 2, 1, false)
    msg_out = f_bp(fill(msg_in, z), wᵢ, ϕᵢ, 2, 1, periodic; showprogress)
    return msg_in, msg_out
end


T = 10
J = 0.5
β = 1/2.0
h = 0.1
m⁰ = 0.1
wᵢ = fill(HomogeneousGlauberFactor(J, h, β), T+1)

using Graphs, IndexedGraphs, Statistics, InvertedIndices, Random
N = 8
rng = MersenneTwister(0)
gg = prufer_decode(rand(rng, 1:N, N-2))
g = IndexedGraph(gg)
ising = Ising(g, fill(J, ne(g)), fill(h, nv(g)), β)
svd_trunc = TruncThresh(1e-3)

Ts = 2:6:40
T = 60
ms_bp = map(Ts) do T
    gl_periodic = Glauber(ising, T)
    bp_periodic = periodic_mpbp(gl_periodic)
    iterate!(bp_periodic; maxiter=20, svd_trunc)
    f(x,i) = -2x+3
    ms_bp = means(f, bp_periodic) |> mean  |> mean
end

ϕᵢ_transient = [ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T ]
gl = Glauber(ising, T, ϕ = fill(ϕᵢ_transient, nv(g)))
bp = mpbp(gl)
iterate!(bp; maxiter=20, svd_trunc)
ms_bp_tr = means(f, bp) |> mean

println("\n")

p_static = [exp(-β*energy(ising, collect(x))) for x in Iterators.product(fill(1:2, nv(g))...)]
p_static ./= sum(p_static)
b_eq = mean([vec(sum(p_static, dims=vertices(g)[Not(i)])) for i in vertices(g)])
m_eq = reduce(-, b_eq)

using Plots
# Ts = 1:3:T
plot(ylabel="avg magnetization", xlabel="t", title="Glauber N=$N")
scatter!(Ts, ms_bp, label="BP periodic")
scatter!(Ts, ms_bp_tr[Ts], label="BP transient")
hline!([m_eq], label="Equilibrium")



# using Graphs, IndexedGraphs, Statistics, InvertedIndices
# g = IndexedGraph(star_graph(z+1))
# ising = Ising(g, fill(J, ne(g)), fill(h, nv(g)), β)
# # gl_transient = Glauber(ising, T; ϕ = fill(ϕᵢ_transient, nv(g)))
# # bp_transient = mpbp(gl_transient)
# # b_transient = exact_marginals(bp_transient)

# ϕᵢ_transient = [ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:60 ]
# gl_transient = Glauber(ising, 60; ϕ = fill(ϕᵢ_transient, nv(g)))
# bp_transient = mpbp(gl_transient)
# iterate!(bp_transient; maxiter=20, svd_trunc=TruncThresh(1e-8))
# f(x,i) = -2x+3

# gl_periodic = Glauber(ising, T; ϕ = fill(ϕᵢ_periodic, nv(g)))
# bp_periodic = periodic_mpbp(gl_periodic)
# # b_periodic = exact_marginals(bp_periodic)
# iterate!(bp_periodic; maxiter=20, svd_trunc=TruncThresh(1e-8))
# f(x,i) = -2x+3


# m_transient = mean(means(f, bp_transient))
# # m_periodic= reduce.(-, mean(b_periodic))
# m_periodic = mean(means(f, bp_periodic))

# using Plots
# scatter(m_transient)
# plot!(m_periodic)
# p_static = [exp(-ising.β*energy(ising, collect(x))) for x in Iterators.product(fill(1:2, nv(g))...)]
# p_static ./= sum(p_static)
# m_eq = reduce(-, mean(vec(sum(p_static, dims=tuple((vertices(g)[Not(i)])...))) for i in vertices(g)))
# hline!([m_eq])




