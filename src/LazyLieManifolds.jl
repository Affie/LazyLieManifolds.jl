module LazyLieManifolds

using LinearAlgebra
using Manifolds
using StaticArrays
using DocStringExtensions

const LLM = LazyLieManifolds

export LLM
##==============================================================================
## Abstract Lie Groups and Manifolds 
##==============================================================================

abstract type AbstractLieGroup end
abstract type AbstractLieAlgebra end

Base.getindex(p::AbstractLieGroup) = p.value
Base.getindex(X::AbstractLieAlgebra) = X.value

# extend Manifolds.jl
Manifolds.affine_matrix(p::AbstractLieGroup) = affine_matrix(manifold(p), p[])
Manifolds.screw_matrix(X::AbstractLieAlgebra) = affine_matrix(manifold(X), X[])


"""
    $(SIGNATURES)
"""
function manifold end

"""
    $(SIGNATURES)
"""
function _identity end

##------------------------------------------------------------------------------
## Extending Manifolds.jl
##------------------------------------------------------------------------------

function Manifolds.vee(X::AbstractLieAlgebra) 
    M = manifold(X)
    return vee(M, make_identity(M, X[]), X[]) #FIXME
end 

# we must use
# seVec = Manifolds.hat(M, p, coords)

function Base.:∘(p::T, q::T) where T <: AbstractLieGroup
    M = manifold(p)
    typeof(p)(compose(M, p[], q[]))
end

function Base.inv(p::AbstractLieGroup)
    M = manifold(p)
    typeof(p)(inv(M, p[]))
end

function Base.:\(p::T, q::T) where T <: AbstractLieGroup
    M = manifold(p)
    typeof(p)(compose(M, inv(M, p[]), q[]))
end

function Base.:/(p::T, q::T) where T <: AbstractLieGroup
    M = manifold(p)
    typeof(p)(compose(M,  p[], inv(M, q[])))
end

function Base.exp(X::AbstractLieAlgebra) #look into retract with RightAction
    M = manifold(X)
    e = identity(M, X[]) #TODO this might not be completely correct
    G = convert(AbstractLieGroup, typeof(X))
    return G(exp(M, e, X[]))
end

function Base.log(p::AbstractLieGroup) #look into inverse_retract with RightAction
    M = manifold(p)
    e = identity(M, p[])
    g = convert(AbstractLieAlgebra, typeof(p))
    return g(log(M, e, p[]))
end

function Base.log(p::T, q::T) where T <: AbstractLieGroup 
    M = manifold(p)
    g = convert(AbstractLieAlgebra, typeof(p))
    return g(log(M, p[], q[]))
end


##==============================================================================
## SpecialEuclidean SE{N} and se{N}
##==============================================================================
export se, SE

"""
    $TYPEDEF

Lie Group representation for `Manifolds.SpecialEuclidean(N)`

Related

[`se`](@ref)
"""
struct SE{N, T} <: AbstractLieGroup
  value::T # Group
end

#TODO assert N and/or SE(p::ProductRepr) = ... 
SE{N}(p::ProductRepr) where N = SE{N, typeof(p)}(p)

SE{N}() where N = _identity(SE{N})

function SE(R::Matrix{T}, t::Vector{T}) where T<:Real
    N = length(t)
    @assert size(R,1) == size(R,2) == N

    _t = SVector{N}(t)
    _R = SMatrix{N,N}(R)

    p = ProductRepr(_t, _R)
    
    return SE{N, typeof(p)}(p)
end

manifold(::SE{N,T}) where {N,T} = SpecialEuclidean(N)


#Default SE{N} representation
function _identity(::Type{SE{N}}, ::Type{T}=Float64) where {N, T}
    t = zeros(SVector{N, T})
    R = SMatrix{N,N,T}(one(T)I)
    SE{N}(ProductRepr(t, R))
end


"""
    $TYPEDEF

Lie algebra representation for `Manifolds.SpecialEuclidean(N)`

Related

[`SE`](@ref)
"""
struct se{N, T} <: AbstractLieAlgebra
  value::T # Tangent Vector (Lie algebra, skew simetric matrix)
end

#TODO assert N and/or se(X::ProductRepr) = ... 
se{N}(X::ProductRepr) where N = se{N, typeof(X)}(X)

function se{N}(X::AbstractVector) where N
    l = length(X)
    @assert manifold_dimension(SpecialEuclidean(N)) == l "X dimension, $l, does not match manifold dimension, $N"
    
    M = SpecialEuclidean(N)
    
    e = _identity(SE{N}, eltype(X))[]
    _X = hat(M, e, X)
        
    return se{N}(_X)
end

manifold(::se{N,T}) where {N,T} = SpecialEuclidean(N)


Base.convert(::Type{AbstractLieGroup}, ::Type{se{N, T}}) where {N,T} = SE{N}
Base.convert(::Type{AbstractLieAlgebra}, ::Type{SE{N, T}}) where {N,T} = se{N}



##------------------------------------------------------------------------------
## other operators
##------------------------------------------------------------------------------
#TODO abstract
function ⊕(p::SE{N}, X::se{N}) where N
    M = manifold(p)
    e = identity(M, p[])
    typeof(p)(compose(M, p[], exp(M, e, X[])))
end

function ⊕(X::se{N}, p::SE{N}) where N
    M = manifold(p)
    e = identity(M, p[])
    typeof(p)(compose(M, exp(M, e, X[]), p[]))
end

# ⊖ - right minus
# X = log(M, e, inv(x) ∘ y)
function ⊖(y::SE{N}, x::SE{N}) where N
    M = manifold(y)
    e = identity(M, y[])
    return log(M, e, inv(x) ∘ y )
end


##==============================================================================
## SpecialOrthogonal SO{N} and so{N}
##==============================================================================
export so, SO

"""
    $TYPEDEF

Lie Group representation for `Manifolds.SpecialOrthogonal(N)`

Related

[`so`](@ref)
"""
struct SO{N, T} <: AbstractLieGroup
  value::T # Group
end

#TODO assert N and/or SO(p::ProductRepr) = ... 
SO{N}(p::StaticMatrix{N,N,T}) where {N, T} = SO{N, typeof(p)}(p)

SO{N}() where N = _identity(SO{N})

function SO(R::Matrix{T}) where T<:Real
    N = size(R,1)
    @assert size(R,1) == size(R,2)

    return SO{N}(SMatrix{N,N}(R))
end

manifold(::SO{N,T}) where {N,T} = SpecialOrthogonal(N)


#Default SO{N} representation
function _identity(::Type{SO{N}}, ::Type{T}=Float64) where {N, T}
    R = SMatrix{N,N,T}(one(T)I)
    SO{N}(R)
end


"""
    $TYPEDEF

Lie algebra representation for `Manifolds.SpecialOrthogonal(N)`

Related

[`SO`](@ref)
"""
struct so{N, T} <: AbstractLieAlgebra
  value::T # Tangent Vector (Lie algebra, skew simetric matrix)
end

#TODO assert N and/or so(X::ProductRepr) = ... 
so{N}(X::StaticMatrix{N,N,T}) where {N,T} = so{N, typeof(X)}(X)

function so{N}(X::AbstractVector) where N
    l = length(X)
    @assert manifold_dimension(SpecialOrthogonal(N)) == l "X dimension, $l, does not match manifold dimension, $N"
    
    M = SpecialOrthogonal(N)
    
    e = _identity(SO{N}, eltype(X))[]
    _X = hat(M, e, X)
        
    return so{N}(_X)
end

manifold(::so{N,T}) where {N,T} = SpecialOrthogonal(N)


Base.convert(::Type{AbstractLieGroup}, ::Type{so{N, T}}) where {N,T} = SO{N}
Base.convert(::Type{AbstractLieAlgebra}, ::Type{SO{N, T}}) where {N,T} = so{N}

##==============================================================================
## RoME like functions sandbox
##==============================================================================

# this is in the "local frame" sequential, a similar operation can be done with the group_log (but together)
function PosePose(ⁱmⱼ::SE{N}, ʷxᵢ::SE{N}, ʷxⱼ::SE{N}) where N

    ʷT̂ⱼ = ʷxᵢ ∘ ⁱmⱼ
    ʲT̂ⱼ = ʷxⱼ \ ʷT̂ⱼ
   
    X = log(ʲT̂ⱼ)

    return vee(X)
end


# this is in the "global frame" e component wise on submanifolds
function PosePose_x(ⁱmⱼ::SE{N}, ʷxᵢ::SE{N}, ʷxⱼ::SE{N}) where N
    M = manifold(ʷxᵢ)

    ʷT̂ⱼ = ʷxᵢ ∘ ⁱmⱼ
    X = se{N}(log(M, ʷT̂ⱼ[], ʷxⱼ[])) # TODO check sign, this (currently) is the vector at ʷT̂ⱼ that points to ʷxⱼ
   
    return vee(X)
end


function Pose2Pose2(meas, wxi, wxj)
    wTjhat = SE2(wxi)*SE2(meas)
    jTjhat = SE2(wxj) \ wTjhat
    return se2vee(jTjhat)
end



end
