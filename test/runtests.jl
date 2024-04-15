using MPSExperiments
using Test

using TensorTrains, Random, ProgressMeter
using TensorTrains.UniformTensorTrains
import TensorTrains.UniformTensorTrains: transfer_operator, leading_eig

using ForwardDiff: gradient
using FiniteDifferences

@testset "Uniform Tensor Train - basics" begin
    include("uniform_tt.jl")
end

@testset "Uniform Tensor Train - derivatives" begin
    include("uniform_tt_derivatives.jl")
end