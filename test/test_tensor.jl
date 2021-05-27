@testset "reshape" begin
    p = randn(2,3)
    x = Var(p)
    y = reshape(x, 6)
    @test reshape(p, 6) == y.data
    gradient!(y)
    @test size(x) == size(x.grad)
end


@testset "matmul" begin
    w = Var([1 2 3; 4 5 6])
    x = Var(transpose(x.data))
    y = w * x
    @test y.data == [14 32; 32 77]

    x = randn(3, 2)
    w = randn(2, 3)
    f = x -> x * Var(w)
    @test gradcheck(f, x)

    x_data = randn(10, 1)
    w_data = randn(1, 5)
    f = w -> matmul(Variable(x_data), w)
    @test gradcheck(f, w_data)
end