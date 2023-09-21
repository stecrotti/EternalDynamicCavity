using TensorCast

function iterate(p, sz; A0 = rand(sz, sz, size(p.tensor)[3:end]...),
        maxiter = 200, tol=1e-4, damp=0.0)
    A = _reshape1(A0)
    Anew = copy(A)
    M = p.tensor
    Mresh = _reshape1(M)
    L = p.L

    for it in 1:maxiter
        q = UniformTensorTrain(A, L)
        normalize!(q)
        G = transfer_operator(q, p)
        E = transfer_operator(q)

        EL = prod(fill(E, L-1)).tensor
        GL = prod(fill(G, L-1)).tensor
        @tullio B[b,a,x] := GL[b,j,a,l] * Mresh[l,j,x]
        
        @cast ELresh[(b,a),(j,l)] := EL[b,j,a,l]
        @cast Bresh[(b,a), x] := B[b,a,x]
        
        for x in axes(Anew, 3)
            k = ELresh \ Bresh[:,x]
            Anew[:,:,x] = reshape(k, Int(sqrt(length(k))), :)'
        end

        ε = norm(A - Anew)
        @show ε
        ε < tol && return A
        A .= damp * A + (1-damp) * Anew 
    end
    return A
end

rng = MersenneTwister(0)

L = 10
qs = (3, 2, 2)
M = rand(rng, 10, 10, qs...)
p = UniformTensorTrain(M, L)
normalize!(p)

sz = 5

A0 = rand(rng, sz, sz, size(p.tensor)[3:end]...)
Aresh = iterate(p, sz; damp=0.8)
A = _reshapeas(Aresh, A0)
q = UniformTensorTrain(A, L)

norm(marginals(q)[1] - marginals(p)[1])