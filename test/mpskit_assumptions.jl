@testset "MPSKit reuse assumptions" begin
    # These pin down the MPSKit behaviors LPTN.jl relies on but does not own,
    # so a future MPSKit release that changes them fails loudly here instead
    # of silently corrupting LPTN tensors/states.

    # GenericMPSTensor{S,3} is 4-legged: Vl ⊗ P ⊗ K ← Vr.
    @test MPSKit.GenericMPSTensor{ComplexSpace, 3} == (AbstractTensorMap{T, ComplexSpace, 3, 1} where {T})

    # FiniteMPS is generic over the site tensor rank, not fixed to MPSTensor.
    P, K, Vmax = ℂ^2, ℂ^2, ℂ^3
    ψ = FiniteMPS(3, P ⊗ K, Vmax)
    @test ψ isa LPTN.FiniteLPTN{ComplexSpace}

    # FiniteMPS stores its gauges as Vector{Union{Missing,A}} fields (.ALs/.ARs/.ACs/.Cs),
    # which LPTN.jl's site accessors read directly via `coalesce` to avoid forcing a
    # gauge transform.
    @test ψ.ALs isa Vector{<:Union{Missing, <:MPSKit.GenericMPSTensor}}
    @test ψ.ARs isa Vector{<:Union{Missing, <:MPSKit.GenericMPSTensor}}
    @test ψ.ACs isa Vector{<:Union{Missing, <:MPSKit.GenericMPSTensor}}
end
