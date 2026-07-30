@testset "LPTNTensor" begin
    Vl, P, K, Vr = ℂ^3, ℂ^2, ℂ^2, ℂ^3
    A = rand(Vl ⊗ P ⊗ K ← Vr)

    @test A isa LPTN.LPTNTensor{ComplexSpace}
    @test numind(A) == 4
    @test space(A, 1) == Vl
    @test lptn_physicalspace(A) == P
    @test krausspace(A) == K
    @test combinedspace(A) == fuse(P, K)

    @test eltype(rand(ComplexF32, Vl ⊗ P ⊗ K ← Vr)) == ComplexF32
    @test eltype(randn(Float64, Vl ⊗ P ⊗ K ← Vr)) == Float64
    @test norm(zeros(Vl ⊗ P ⊗ K ← Vr)) == 0
end
