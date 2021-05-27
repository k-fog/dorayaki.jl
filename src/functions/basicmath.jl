import Base: sin, cos, tanh, exp, log

"""
    Sin <: Func
"""
@func mutable struct Sin end

foward(f::Sin, x) = sin.(x)

backward(f::Sin, gy) = cos.(gy)

sin(x::Var) = Sin()(x)


"""
    Cos <: Func
"""
@func mutable struct Cos end

foward(f::Cos, x) = cos.(x)

backward(f::Cos, gy) = -sin.(gy)

cos(x::Var) = Cos()(x)


"""
    Tanh <: Func
"""
@func mutable struct Tanh end

forward(f::Tanh, x) = tanh.(x)

function backward(f::Tanh, gy)
    y = f.outputs[1].value
    return gy .* (1 - y .* y)
end

tanh(x::Var) = Tanh()(x)


"""
    Exp <: Func
"""
@func mutable struct Exp end

forward(f::Exp, x) = exp.(x)

backward(f::Exp, gy) = gy .* f.outputs[1].value

exp(x::Var) = Exp()(x)


"""
    Log <: Func
"""
@func mutable struct Log end

forward(f::Log, x) = log.(x)

function backward(f::Log, gy)
    x, = f.args
    return gy / x
end

log(x::Var) = Log()(x)