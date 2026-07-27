"""
    FiniteLPTN{S} = MPSKit.FiniteMPS{LPTNTensor{S}, MPSKit.MPSBondTensor{S}}

A finite chain of [`LPTNTensor`](@ref)s. Since `LPTNTensor` is literally an
`MPSKit.GenericMPSTensor{S,3}`, this reuses `MPSKit.FiniteMPS`'s lazily computed
left/right/center gauges as-is. Note that `Tr[ρ] = norm(ψ)^2`, not `norm(ψ)` itself: tracing
the purified density operator sums `|Λ|^2` over both physical and Kraus indices, i.e. the
*squared* 2-norm of `ψ` as a plain vector. See [`lptn_trace`](@ref).
"""
const FiniteLPTN{S} = MPSKit.FiniteMPS{<:LPTNTensor{S}, <:MPSKit.MPSBondTensor{S}}

"""
    FiniteLPTN([f, eltype], N::Int, P::S, K::S,
               maxvirtualspace::Union{S,Vector{S}}; kwargs...) where {S<:ElementarySpace}

Construct a [`FiniteLPTN`](@ref) of `N` sites with physical space `P` and Kraus space `K`
at every site, by forwarding to `MPSKit.FiniteMPS` with `P` and `K` fused into a single
composite physical space.

### Keywords
- `normalize=true`: normalize the constructed state
- `left=unitspace(S)`: left-most virtual space
- `right=unitspace(S)`: right-most virtual space
"""
function FiniteLPTN(
        f, elt, N::Int, P::S, K::S, maxvirtualspace::Union{S, Vector{S}}; kwargs...
    ) where {S <: ElementarySpace}
    return MPSKit.FiniteMPS(f, elt, N, P ⊗ K, maxvirtualspace; kwargs...)
end
function FiniteLPTN(
        N::Int, P::S, K::S, maxvirtualspace::Union{S, Vector{S}}; kwargs...
    ) where {S <: ElementarySpace}
    return MPSKit.FiniteMPS(N, P ⊗ K, maxvirtualspace; kwargs...)
end

"""
    lptn_physicalspace(ψ::FiniteLPTN, n::Integer)

Return the physical space at site `n` of the LPTN chain `ψ`, without forcing computation
of any gauge that isn't already cached.
"""
function lptn_physicalspace(ψ::FiniteLPTN, n::Integer)
    return lptn_physicalspace(coalesce(ψ.ALs[n], ψ.ARs[n], ψ.ACs[n]))
end

"""
    krausspace(ψ::FiniteLPTN, n::Integer)

Return the Kraus (purification) space at site `n` of the LPTN chain `ψ`, without forcing
computation of any gauge that isn't already cached.
"""
function krausspace(ψ::FiniteLPTN, n::Integer)
    return krausspace(coalesce(ψ.ALs[n], ψ.ARs[n], ψ.ACs[n]))
end

"""
    lptn_trace(ψ::FiniteLPTN)

Return `Tr[ρ]`, the trace of the density operator purified by the LPTN chain `ψ`. This is
`norm(ψ)^2`, not `norm(ψ)`: tracing `ρ` sums `|Λ|^2` over both physical and Kraus indices,
i.e. the squared 2-norm of `ψ` as a plain vector. Computed as `norm(ψ)^2` rather than
`dot(ψ, ψ)` so it reuses `MPSKit.FiniteMPS`'s O(1) gauge-center shortcut instead of a full
O(N) chain contraction.
"""
lptn_trace(ψ::FiniteLPTN) = norm(ψ)^2
