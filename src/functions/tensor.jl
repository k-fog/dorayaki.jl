import Base: reshape, adjoint, transpose, *, sum

export matmul, linear, reshape, transpose, adjoint, broadcastto, sumto


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
    Linear_F <: Func
"""
@func mutable struct Linear_F end

forward(f::Linear_F, w, x, b=nothing) = b isa Nothing ? w * x : w * x .+ b

function backward(f::Linear_F, gy)
    w, x, b = length(f.args) == 3 ? f.args : (f.args..., nothing)
    gb = b isa Nothing ? nothing : sumto(gy, size(b))
    gw = matmul(gy, transpose(x))
    gx = matmul(transpose(w), gy)
    return gw, gx, gb
end

linear(w, x, b=nothing) = b isa Nothing ? Linear_F()(w, x) : Linear_F()(w, x, b)


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
    if length(shape) == 1 && shape[1] isa Union{Tuple,AbstractArray}
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

transpose(x::Var) = Transpose()(x)


"""
    Sum <: Func
"""
@func mutable struct Sum
    dims::Union{Int,Tuple,Nothing}
    Sum(dims) = new(dims)
end

function forward(f::Sum, x)
    if f.dims isa Nothing
        return sum(x)
    else
        return sum(x, dims=f.dims)
    end
end

backward(f::Sum, gy) = broadcastto(gy, size(args[1]))

sum(x::Var; dims=nothing) = Sum(dims)(x)


"""
    BroadcastTo <: Func
"""
@func mutable struct BroadcastTo
    shape::Tuple
    BroadcastTo(shape) = new(shape)
end

forward(f::BroadcastTo, x) = x .* ones(f.shape)

backward(f::BroadcastTo, gy) = sumto(gy, size(f.args[1]))

broadcastto(x::Var, shape) = size(x) == shape ? asvar(x) : BroadcastTo(shape)(x)


"""
    SumTo <: Func
"""
@func mutable struct SumTo
    shape::Tuple
    SumTo(shape) = new(shape)
end

forward(f::SumTo, x) = _sumto(x, f.shape)

backward(f::SumTo, gy) = broadcastto(gy, size(f.args[1]))

sumto(x::Var, shape) = size(x) == shape ? asvar(x) : SumTo(shape)(x)