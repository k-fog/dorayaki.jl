using Dorayaki
using Random
using Plots
using BenchmarkTools

Random.seed!(0)

const lr = 0.1
const iters = 100

x = rand(1, 100)
y = 5 .+ 2 .* x .+ rand(1, 100)

function main(x, y) 
    W = Var(zeros(1, 1))
    b = Var(zeros(1))

    predict(x) = W * x + b

    for i in 1:iters
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        cleargrad!(W)
        cleargrad!(b)

        gradient!(loss)
        W.data .-= lr .* W.grad.data
        b.data .-= lr .* b.grad.data
        # println(W.data, b.data, loss.data)
    end
    return W.data[1], b.data[1]
end

ret = @benchmark main(x, y)
#= 
f(a) = ret[1] * a + ret[2]
fig = plot(x, y, st=:scatter)
plot!(fig, f) =#