using Dorayaki
using Random
using Plots

const lr = 0.2
const iters = 10000

function main(x, y)
    I, H, O = 1, 10, 1
    l1 = Linear(I, H)
    l2 = Linear(H, O)

    function predict(p)
        q = l1(p)
        q = sigmoid(q)
        q = l2(q)
        return q
    end

    for i in 1:iters
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        cleargrad!(l1, l2)

        backward!(loss)

        for l in (l1, l2)
            for p in params(l)
                p.data .-= lr .* p.grad.data
            end
        end

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