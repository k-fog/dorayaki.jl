import Base: reshape, adjoint, transpose, *

export matmul, reshape, transpose, adjoint


"""
    MatMul <: Func
"""
@func mutable struct MatMul end

forward(f::MatMul, w, x) = w * x

function backward(f::MatMul, gy)
    w, x = f.args
    gw = matmul(gy, transpose(x))
    gx = matmul(transpose(w), gy)
    return gw, gx
end

matmul(w, x) = MatMul()(w, x)

Base.:*(A::Var, B::Var) = matmul(A, B)
Base.:*(A::Var, B) = matmul(A, B)
Base.:*(A, B::Var) = matmul(A, B)


"""
    Reshape <: Func
"""
@func mutable struct Reshape
    shape::Tuple
    Reshape(shape) = new(shape)
end

forward(f::Reshape, x) = reshape(x, f.shape)

backward(f::Reshape, gy) = reshape(gy, size(f.args[1]))

function reshape(x, shape...)
    if length(shape) == 1 && shape[1] isa Union{Tuple, AbstractArray}
        shape = shape[1]
    end
    size(x) == shape && return asvar(x)
    return Reshape(shape)(x)
end

adjoint(x::Var) = transpose(x)


"""
    transpose <: Func
"""
@func mutable struct Transpose end

forward(f::Transpose, x) = transpose(x)

backward(f::Transpose, gy) = transpose(gy)

transpose(x) = Transpose()(x)