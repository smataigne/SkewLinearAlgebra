module SkewLinearAlgebraChainRulesCoreExt

using LinearAlgebra
using SkewLinearAlgebra
using ChainRulesCore


function ChainRulesCore.rrule(::Type{SkewHermitian}, val) 
    y = SkewHermitian(val)
    function Foo_pb(ΔFoo) 
        if isa(ΔFoo, SkewHermitian)
            return NoTangent(), unthunk(ΔFoo).data
        else
            return (NoTangent(), unthunk(ΔFoo))
        end
    end
    return y, Foo_pb
end

function ChainRulesCore.rrule(::typeof(pfaffian), A::SkewHermitian)
    Ω = pfaffian(A)
    pfaffian_pullback(ΔΩ) = NoTangent(), SkewHermitian(rmul!(inv(A)', dot(Ω, ΔΩ))) #potentially need the 0.5 here !
    return Ω, pfaffian_pullback
end

function ChainRulesCore.ProjectTo{<:SkewHermitian}(A::AbstractArray)
    return skewhermitian(A)
end

end
