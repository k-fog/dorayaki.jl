import Base: reshape, transpose

export reshape, transpose

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


"""
    transpose <: Func
"""
@func mutable struct Transpose end

forward(f::Transpose, x) = transpose(x)

backward(f::Transpose, gy) = transpose(gy)

transpose(x) = Transpose()(x)