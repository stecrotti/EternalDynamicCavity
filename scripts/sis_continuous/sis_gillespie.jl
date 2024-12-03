using Graphs, IndexedGraphs, Statistics
using MatrixProductBP, MatrixProductBP.Models
using JLD2


T = 600        # final time
k = 3         # degree
λs = [0.1, 0.2, 0.3, 0.4]      # rate of transmission
ρ = 0.1       # rate of recovery
γ = 0.9      # prob. of zero patient

nsamples = 5*10^3

Ns = [100, 500, 2000]

seed = 0

ps_gillespie = map(λs) do λ
    ps = map(Ns) do N
        gg = random_regular_graph(N, k; seed)
        g = IndexedGraph(gg)

        sis = SIS(g, λ, ρ, T; γ)
        p_gill,_ = continuous_sis_sampler(sis, T, λ, ρ; nsamples, sites=1:N,
            Δt=1e-1, discard_dead_epidemics=false)
        m_gill = mean(p_gill)

        println("\nFinished N=$N\n")

        Tmean = T ÷ 4
        p_gillespie = mean(m_gill[end-Tmean:end])
    end
    println("\nFinished λ=$λ\n")
    ps
end

jldsave((@__DIR__)*"/../../data/sis_gillespie.jld2"; k, λs, ρ, ps_gillespie, Ns, nsamples)