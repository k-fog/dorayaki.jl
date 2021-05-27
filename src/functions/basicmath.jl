import Base: sin, cos

"""
    Sin <: Func
"""
@func mutable struct Sin end

foward(f::Sin, x) = sin.(x)

backward(f::Sin, gy) = cos.(gy)