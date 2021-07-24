export Sigmoid, sigmoid


"""
    Sigmoid <: Func
"""
@func mutable struct Sigmoid end

forward(f::Sigmoid, x) = @. tanh(x * 0.5) * 0.5 + 0.5

function backward(f::Sigmoid, gy)
    y = f.outputs[1]
    return gy .* y .* (1 - y)
end

sigmoid(x) = Sigmoid()(x)