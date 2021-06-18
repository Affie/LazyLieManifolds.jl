using LinearAlgebra: isapprox
using LazyLieManifolds
using Test
using StaticArrays
using LinearAlgebra

import LazyLieManifolds: ⊕

using Manifolds

@testset "LazyLieManifolds.jl" begin
         
    _p0 = 1.0I(3)
    _X1 = [0.0 -pi/2 1.0;  pi/2 0.0 0.0; 0.0 0.0 1.0]
    _X2 = [0.0  pi/2 1.0; -pi/2 0.0 0.0; 0.0 0.0 1.0]
    
    p0 = SE([1.0 0.0; 0 1.0], [0.0, 0.0])

    M = LLM.manifold(p0)
    @test all(isapprox.(_p0, affine_matrix(M, p0[])))
    
    X1 = se{2}([1., 0, pi/2])
    X2 = se{2}([1., 0, -pi/2])
    
    @test all(isapprox.(_X1, affine_matrix(M, X1[])))
    @test all(isapprox.(_X2, affine_matrix(M, X2[])))

    p1 = p0 ⊕ X1
    
    p2 = p1 ⊕ X2
    
    p3 = p1 ∘ p2
    
    @time LLM.PosePose(p1, p0, p2)
    @time LLM.PosePose_x(p1, p0, p2)
    

    SE{3}()
    
    p0 = LLM._identity(SE{3})
    X1 = se{3}([1., 0, 0, pi/2, 0, 0])
    X2 = se{3}([1., 0, 0, -pi/2, 0, 0])
    p1 = p0 ⊕ X1
    p2 = p1 ⊕ X2
    
    # @code_warntype exp(X1)
    log(p2)
    log(p1, p2)
    
    
    ##
    
    p0 = SO([1.0 0.0; 0 1.0])
    X1 = so{2}([pi/2])
    X2 = se{2}([1., 0, -pi/2])
    
    # p1 = p0 ⊕ X1
    # p2 = p1 ⊕ X2
    # p3 = p1 ∘ p2
    
    SO{5}()
    
    p0 = LLM._identity(SO{3})
    X1 = so{3}([pi/2, 0, 0])
    X2 = so{3}([-pi/2, 0, 0])
    
    log(p2)
    log(p1, p2)

end
