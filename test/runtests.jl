using Test
using LPTN
using TensorKit
using MPSKit

@testset "LPTN.jl" begin
    include("tensors.jl")
    include("states.jl")
    include("mpskit_assumptions.jl")
end
