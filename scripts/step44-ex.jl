using Dorayaki
using Random
using Plots

const lr = 0.2
const iters = 10000

function main(x, y)
    I, H, O = 1, 10, 1
    model = Chain(Linear(I, H), Sigmoid(), Linear(H, O))

    for i in 1:iters
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)

        cleargrad!(model)

        gradient!(loss)

        for l in params(model)
            p.data .-= lr .* p.grad.data
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