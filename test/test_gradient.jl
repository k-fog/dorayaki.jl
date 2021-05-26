sphere(x, y) = x^2 + y^2
matyas(x, y) = 0.26 .* (x^2 + y^2) - 0.48 .* x .* y
goldstein(x, y) = (1 + (x + y + 1)^2 .* (19 - 14 .* x + 3 .* x^2 - 14 .* y + 6 .* x .* y + 3 .* y^2)) .*
    (30 + (2 .* x - 3 .* y)^2 .* (18 - 32 .* x + 12 .* x^2 + 48 .* y - 36 .* x .* y + 27 .* y^2))

function check(f, ans)
    x = Var(1)
    y = Var(1)
    z = f(x, y)
    gradient!(z)
    (x.grad.data, y.grad.data) == ans
end

@test check(sphere, ([2.0], [2.0]))
@test check(matyas, ([0.040000000000000036], [0.040000000000000036]))
@test check(goldstein, ([-5376.0], [8064.0]))