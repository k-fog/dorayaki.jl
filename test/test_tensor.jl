@testset "reshape" begin
    p = randn(2,3)
    x = Var(p)
    y = reshape(x, 6)
    @test reshape(p, 6) == y.data
    gradient!(y)
    @test size(x) == size(x.grad)
end