using Dorayaki
using Test

@testset "Add" begin
    x1 = [1,2,3]
    x2 = Var([1,2,3])
    y = x1 + x2
    @test y.data == [2,4,6]
end

@testset "Pow" begin
    x = Var(2)
    @test (x^10).data == [1024]
end