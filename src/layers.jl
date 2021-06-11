export Chain, applychain, params, Linear, MLP

abstract type AbstractLayer <: Func end

"""
    Chain
"""
struct Chain
    layers::Tuple
    Chain(xs...) = new(xs)
end

applychain(layers::Tuple, x...) = applychain(layers[2:end], first(layers)(x...))
applychain(layers::Tuple{}, x...) = length(x) > 1 ? x : x[1]

(c::Chain)(xs...) = applychain(c.layers, xs...)

Base.getindex(c::Chain, i::Int) = c.layers[i]
Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

function (layer::AbstractLayer)(args...)
    outputs = forward(layer, args...)
    outputs isa Tuple || (outputs = (outputs,))
    layer.args = Tuple(WeakRef(x) for x in args)
    layer.outputs = Tuple(WeakRef(y) for y in outputs)
    return length(outputs) > 1 ? outputs : outputs[1]
end


function params(layer::AbstractLayer)
    
end


function cleargrad!(x::AbstractLayer)
    for p in params(x)
        cleargrad!(p)
    end
end

cleargrad!(x::AbstractLayer...) = cleargrad!.(x)


"""
    Linear <: AbstractLayer
"""
@func mutable struct Linear <: AbstractLayer
    weight::Var
    bias::Union{Var,Nothing}

    function Linear(in, out, nobias=false, dtype=Float32)
        w = Param(_initW(in, out, dtype))
        b = nobias ? nothing : Param(zeros(dtype, out))
        return new(w, b)
    end
end

_initW(in, out, dtype) = randn(dtype, out, in) .* dtype(sqrt(1 / in))

function forward(layer::Linear, x)
    return linear(layer.weight, x, layer.bias)
end
