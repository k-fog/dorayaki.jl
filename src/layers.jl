#=
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

Base.length(c::Chain) = length(c.layers)
Base.getindex(c::Chain, i::Int) = c.layers[i]
Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)
Base.lastindex(c::Chain) = c.layers[end]

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

function (layer::AbstractLayer)(args...)
    outputs = forward(layer, args...)
    outputs isa Tuple || (outputs = (outputs,))
    layer._inputs = Tuple(WeakRef(x) for x in args)
    layer._outputs = Tuple(WeakRef(y) for y in outputs)
    return length(outputs) > 1 ? outputs : outputs[1]
end

vars(layer::AbstractLayer) = [topsort(n.value) for n in layer.outputs]

vars(model::Chain) = vars(model[end])

function params(layer::AbstractLayer)
    nodes = vars(layer)
    filter(x -> isparam(x), nodes)
end

params(model::Chain) = params(model[end])


cleargrad!(model::Chain) = cleargrad!(vars(model))

cleargrad!(layer::AbstractLayer) = cleargrad!(vars(layer))

cleargrad!(layers::AbstractLayer...) = cleargrad!.(layers)


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


=#