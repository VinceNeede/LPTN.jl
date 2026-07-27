@testset "LPTNTensor / LPTNMapSpace" begin
    Vl, P, K, Vr = ℂ^3, ℂ^2, ℂ^2, ℂ^3
    A = rand(LPTN.LPTNMapSpace(Vl, P, K, Vr))

    @test A isa LPTN.LPTNTensor{ComplexSpace}
    @test numind(A) == 4
    @test space(A, 1) == Vl
    @test lptn_physicalspace(A) == P
    @test krausspace(A) == K
    @test combinedspace(A) == fuse(P, K)

    @test eltype(rand(LPTN.LPTNMapSpace(Vl, P, K, Vr))) == Defaults.eltype
    @test eltype(rand(ComplexF32, LPTN.LPTNMapSpace(Vl, P, K, Vr))) == ComplexF32
    @test eltype(randn(Float64, LPTN.LPTNMapSpace(Vl, P, K, Vr))) == Float64
    @test norm(zeros(LPTN.LPTNMapSpace(Vl, P, K, Vr))) == 0
end
