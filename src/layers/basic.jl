"""
    Linear <: AbstractLayer
"""
@func mutable struct Linear <: AbstractLayer
    weight::Var
    bias::Union{Var,Nothing}

    function Linear(in, out, nobias=false, dtype=Float32)
        w = Var(_initW(in, out, dtype))
        b = nobias ? nothing : Var(zeros(dtype, out))
        return new(w, b)
    end
end

_initW(in, out, dtype) = randn(dtype, out, in) .* dtype(sqrt(1 / in))

function forward(layer::Linear, x)
    return linear(layer.weight, x, layer.bias)
end