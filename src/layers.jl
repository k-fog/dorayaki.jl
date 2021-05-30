export params, Linear

abstract type AbstractLayer <: Func end


function (lyr::AbstractLayer)(args...)
    outputs = forward(lyr, args...)
    outputs isa Tuple || (outputs = (outputs,))
    lyr.args = Tuple(WeakRef(x) for x in args)
    lyr.outputs = Tuple(WeakRef(y) for y in outputs)
    return length(outputs) > 1 ? outputs : outputs[1]
end


function params(lyr::AbstractLayer)
    pnames = propertynames(lyr)
    ps = []
    for pname in pnames
        pname in (:args, :outputs) && continue
        p = isdefined(lyr, pname) ? getfield(lyr, pname) : nothing
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
    Layer <: AbstractLayer
"""
@func mutable struct Layer <: AbstractLayer
end


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

function forward(lyr::Linear, x)
    return linear(lyr.weight, x, lyr.bias)
end

function Base.show(io::IO, l::Linear)
  print(io, "Linear(", size(l.weight, 2), ", ", size(l.weight, 1))
  l.bias isa Nothing && print(io, "; bias=false")
  print(io, ")")
end