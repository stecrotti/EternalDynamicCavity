# generate a random bipartite regular graph
function random_bipartite_regular(nA, nB, kA, kB;
    accept_multiedges=false, rng=Random.default_rng(), ntrials = 10^3)

    nedges = nA * kA
    @assert nB * kB == nedges

    neigsA = [Int[] for _ in 1:nA]
    neigsB = [Int[] for _ in 1:nB]
    edgesleft = zeros(Int, nedges)
    edgesright = zeros(Int, nedges)

    multi_edge_found = false
    for _ in 1:ntrials
        multi_edge_found = false
        neigsA .= [Int[] for _ in 1:nA]
        neigsB .= [Int[] for _ in 1:nB]
        edgesleft .= 0
        edgesright .= 0

        for iA in 1:nA
            edgesleft[kA*(iA-1)+1:kA*iA] .= iA
        end

        shuffle!(rng, edgesleft)   # Permute nodes on the left

        for iB in 1:nB
            for c in kB*(iB-1)+1:kB*iB
                iA = edgesleft[c]
                if isempty(findall(isequal(iB), neigsA[iA]))
                    push!(neigsA[iA], iB)
                    push!(neigsB[iB], iA)
                elseif !accept_multiedges
                    multi_edge_found=true
                    break
                end
            end
        end

        if !multi_edge_found 
            I = reduce(vcat, fill(iA, kA) for iA in 1:nA)
            J = reduce(vcat, neigsA)
            K = ones(Int, nedges)
            return sparse(I, J, K)
        end
    end
    error("Could not build graph")
end