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
combinedspace(A::LPTNTensor) = fuse(space(A, 2), space(A, 3))
