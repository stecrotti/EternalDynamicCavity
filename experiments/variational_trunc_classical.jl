using MPSExperiments
import TensorTrains: InfiniteUniformTensorTrain, marginals
using Random
using GenericSchur

rng = MersenneTwister(0)
D = 5
d = D
M = rand(rng, BigFloat, D, D, 2, 3)
p = InfiniteUniformTensorTrain(M)
q = truncate_eachtensor(p, d)

only(marginals(q)) - only(marginals(p))

# q = truncate_variational(p, d)