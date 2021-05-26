using Dorayaki
using Test

@testset "functions" begin
    include("test_arithmetic.jl")
    include("test_tensor.jl")
end

@testset "gradient" begin
    include("test_gradient.jl")
end
