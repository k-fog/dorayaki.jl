using Dorayaki
using Dorayaki: Functions as F
using Random
using Plots

const lr = 0.2
const iters = 10000

function main(x, y)
    I, H, O = 1, 10, 1
    w1 = Var(0.01 * randn(H, I))
    b1 = Var(zeros(H))
    w2 = Var(0.01 * randn(O, H))
    b2 = Var(zeros(O))

    function predict(p)
        q = F.linear(w1, p, b1)
        q = F.sigmoid(q)
        q = F.linear(w2, q, b2)
        return q
    end

    for i in 1:iters
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        cleargrad!(w1, b1, w2, b2)

        backward!(loss)

        w1.data .-= lr .* w1.grad.data
        b1.data .-= lr .* b1.grad.data
        w2.data .-= lr .* w2.grad.data
        b2.data .-= lr .* b2.grad.data

        if i % 1000 == 0
            println(loss.data)
        end
    end
    return predict
end

x = rand(1, 100)
y = sin.(2 * pi .* x) .+ rand(1, 100)

ret = @time main(x, y)
fig = plot(x', y', st=:scatter)
f(x) = ret(x).data[1]
plot!(fig, f)