export sigmoid


"""
    Sigmoid <: Func
"""
@func mutable struct Sigmoid end

forward(f::Sigmoid, x) = @. tanh(x * 0.5) * 0.5 + 0.5

function backward(self, gy)
    y = self.outputs[1].value
    return gy .* y .* (1 - y)
end

sigmoid(x) = Sigmoid()(x)