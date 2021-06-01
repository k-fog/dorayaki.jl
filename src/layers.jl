import Base: show

export Chain, applychain, params, Linear, MLP

abstract type AbstractLayer <: Func end

struct Chain
    layers::Tuple
    Chain(xs...) = new(xs)
end

applychain(layers::Tuple, x...) = applychain(tail(layers), first(layers)(x...))

(c::Chain)(xs...) = applychain(c.layers, xs...)

function show(io::IO, c::Chain)
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
    pnames = propertynames(layer)
    ps = []
    for pname in pnames
        pname in (:args, :outputs) && continue
        p = isdefined(layer, pname) ? getfield(layer, pname) : nothing
        if isparam(p)
            push!(ps, p)
        elseif p isa AbstractLayer
            append!(ps, params(p))
        end
    end
    return ps
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

function show(io::IO, l::Linear)
  print(io, "Linear(", size(l.weight, 2), ", ", size(l.weight, 1))
  l.bias isa Nothing && print(io, "; bias=false")
  print(io, ")")
end


"""
    MLP <: AbstractLayer
"""
@func mutable struct MLP
    
end