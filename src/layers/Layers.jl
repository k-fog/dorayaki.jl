module Layers

using ..Dorayaki

abstract type AbstractLayer <: Func end

function (layer::AbstractLayer)(args...)
    outputs = forward(layer, args...)
    outputs isa Tuple || (outputs = (outputs,))
    layer._inputs = Tuple(WeakRef(x) for x in args)
    layer._outputs = Tuple(WeakRef(y) for y in outputs)
    return length(outputs) > 1 ? outputs : outputs[1]
end



include("basic.jl")

end