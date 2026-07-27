"""
    LPTNTensor{S} = MPSKit.GenericMPSTensor{S, 3}

A single-site tensor of a locally purified tensor network (LPTN), with four indices:
a left virtual leg, a physical leg, a Kraus (purification) leg, and a right virtual leg,
of the form `Vₗ ⊗ P ⊗ K ← Vᵣ`.
"""
const LPTNTensor{S} = MPSKit.GenericMPSTensor{S, 3}

"""
    lptn_physicalspace(A::LPTNTensor)

Return the physical space of the LPTN tensor `A`, i.e. its second leg.
"""
lptn_physicalspace(A::LPTNTensor) = space(A, 2)

"""
    krausspace(A::LPTNTensor)

Return the Kraus (purification) space of the LPTN tensor `A`, i.e. its third leg.
"""
krausspace(A::LPTNTensor)         = space(A, 3)

"""
    combinedspace(A::LPTNTensor)

Return the fusion of the physical and Kraus spaces of the LPTN tensor `A`.
"""
combinedspace(A::LPTNTensor)      = fuse(space(A, 2), space(A, 3))

"""
$(TYPEDEF)

Description of the space `Vₗ ⊗ P ⊗ K ← Vᵣ` of an [`LPTNTensor`](@ref), used to construct
tensors of a given shape via [`rand`](@ref), [`randn`](@ref), and [`zeros`](@ref).

### Fields
$(TYPEDFIELDS)
"""
struct LPTNMapSpace{S<:ElementarySpace}
    "left virtual space"
    Vₗ::S
    "physical space"
    P::S
    "Kraus (purification) space"
    K::S
    "right virtual space"
    Vᵣ::S
end

# --- constructors ---
const _LPTNMAPSPACE_FILL_DESCRIPTIONS = Dict(
    :rand => "filled with uniformly distributed random entries",
    :randn => "filled with normally distributed random entries",
    :zeros => "filled with zeros",
)

for f in (:rand, :randn, :zeros)
    fill_description = _LPTNMAPSPACE_FILL_DESCRIPTIONS[f]
    @eval begin
        @doc """
            $($f)([T::Type=$(Defaults.eltype)], A::LPTNMapSpace)

        Construct a tensor with `eltype` `T` and spaces `A.Vₗ ⊗ A.P ⊗ A.K ← A.Vᵣ`, $($fill_description).
        """
        function Base.$f(::Type{T}, A::LPTNMapSpace) where {T}
            return $f(T, A.Vₗ ⊗ A.P ⊗ A.K ← A.Vᵣ)
        end
        Base.$f(A::LPTNMapSpace) = $f(Defaults.eltype, A)
    end
end
