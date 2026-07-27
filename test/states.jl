@testset "FiniteLPTN" begin
    N = 4
    P, K, Vmax = ℂ^2, ℂ^2, ℂ^3
    ψ = FiniteLPTN(N, P, K, Vmax)

    @test ψ isa LPTN.FiniteLPTN{ComplexSpace}
    @test length(ψ) == N
    for n in 1:N
        @test lptn_physicalspace(ψ, n) == P
        @test krausspace(ψ, n) == K
    end
    @test norm(ψ) ≈ 1.0
    @test lptn_trace(ψ) ≈ 1.0

    # site accessors must read whatever gauge is already cached, without
    # forcing computation of AC (see LPTN.jl/src/states.jl)
    cached_before = .!ismissing.(ψ.ACs)
    lptn_physicalspace(ψ, 2)
    krausspace(ψ, 2)
    @test (.!ismissing.(ψ.ACs)) == cached_before
end

@testset "lptn_trace vs norm" begin
    # normalize=true forces norm(ψ) == 1, and 1^2 == 1, so a normalized state can't tell
    # `lptn_trace` and `norm` apart. Use an unnormalized state, and cross-check against a
    # brute-force trace of the purified density operator (sum |Λ|^2 over every physical
    # and Kraus index), which is the actual definition of Tr[ρ].
    P, K, Vb = ℂ^2, ℂ^2, ℂ^2
    A1 = rand(LPTN.LPTNMapSpace(oneunit(Vb), P, K, Vb))
    A2 = rand(LPTN.LPTNMapSpace(Vb, P, K, oneunit(Vb)))
    ψ = FiniteMPS([A1, A2])

    @test lptn_trace(ψ) ≈ norm(ψ)^2
    @test !isapprox(lptn_trace(ψ), norm(ψ))

    a1, a2 = convert(Array, A1), convert(Array, A2)
    dP, dK, dVb = dim(P), dim(K), dim(Vb)
    Λ = zeros(ComplexF64, dP, dK, dP, dK)
    for p1 in 1:dP, k1 in 1:dK, p2 in 1:dP, k2 in 1:dK, b in 1:dVb
        Λ[p1, k1, p2, k2] += a1[1, p1, k1, b] * a2[b, p2, k2, 1]
    end
    @test lptn_trace(ψ) ≈ sum(abs2, Λ)
end
