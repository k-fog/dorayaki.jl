using Dorayaki
using Test

@testset "Dorayaki.jl" begin
    @testset "basicmath" begin
        include("basicmath.jl")
    end
end
