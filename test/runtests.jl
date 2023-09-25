using MPSExperiments
using Test

using TensorTrains, Random, ProgressMeter

@testset "Uniform Tensor Train - basics" begin
    include("uniform_tt.jl")
end

@testset "Uniform Tensor Train - derivatives" begin
    include("uniform_tt_derivatives.jl")
end
