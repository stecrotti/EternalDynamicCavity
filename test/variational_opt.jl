using TensorCast

L = 10
qs = (3, 2)
M = rand(rng, 6, 6, qs...)
# EG = transfer_operator(UniformTensorTrain(M, L))
# λ = real(eig(EG)[:λ])
# M ./= sqrt(λ)
p = UniformTensorTrain(M, L)
normalize!(p)

sz = (3, 3)

function iterate(p, sz; A0 = rand(sz..., size(p.tensor)[3:end]...),
        maxiter = 200, tol=1e-4, damp=0.0)
    A = A0
    Anew = copy(A)
    M = p.tensor
    L = p.L

    for it in 1:maxiter
        q = UniformTensorTrain(A, L)
        # G = transfer_operator(q, p)
        # λ = real(eig(G)[:λ])
        # A ./= λ
        # q = UniformTensorTrain(A, L)
        normalize!(q)
        G = transfer_operator(q, p)
        E = transfer_operator(q)

        EL = prod(fill(E, L-1)).tensor
        GL = prod(fill(G, L-1)).tensor
        @tullio B[b,a,x,y] := GL[b,j,a,l] * M[l,j,x,y]
        
        @cast ELresh[(b,a),(j,l)] := EL[b,j,a,l]
        @cast Bresh[(b,a), x, y] := B[b,a,x,y]
        
        for x in 1:qs[1]
            for y in 1:qs[2]
                k = ELresh \ Bresh[:,x,y]
                Anew[:,:,x,y] = reshape(k, Int(sqrt(length(k))), :)'
            end
        end

        ε = norm(A - Anew)
        @show ε
        ε < tol && return A
        A .= damp * A + (1-damp) * Anew 
    end
    return A
end

A = iterate(p, sz; damp=0.5)
q = UniformTensorTrain(A, L)

norm(marginals(q)[1] - marginals(p)[1])

