@testset "add" begin
    x1 = [1,2,3]
    x2 = Var([1,2,3])
    y = x1 + x2
    @test y.data == [2,4,6]
    
    x = randn(3, 3)
    y = randn(3, 3)
    f = x -> x .+ y
    @test gradcheck(f, x)

    x = Var(randn(3, 3))
    y = randn(3)
    @test gradcheck(f, x)
end


@testset "mul" begin
    x1 = [1, 2, 3]
    x2 = Var([1, 2, 3])
    y = x1 .* x2
    @test y.data == [1, 4, 9]

    x = randn(3, 3)
    y = randn(3, 3)
    f = x -> x .* y
    @test gradcheck(f, x)

    y = randn(3, 1)
    @test gradcheck(f, x)

    y = randn(3)
    @test gradcheck(f, x)

    f = y -> x .* y
    @test gradcheck(f, x)
end


@testset "div" begin
    x1 = [1, 2, 3]
    x2 = Var([1, 2, 3])
    y = x1 / x2
    @test y.data == [1, 1, 1]

    x = randn(3, 3)
    y = randn(3, 3)
    f = x -> x ./ y
    @test gradcheck(f, x)
    y = randn(3)
    @test gradcheck(f, x)
end


@testset "pow" begin
    x = Var(2)
    @test (x^10).data == [1024]
    y = x^2
    backward!(y)
    @test x.grad.data == [4.0]
end