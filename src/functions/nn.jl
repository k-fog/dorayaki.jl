module NN

using ..Dorayaki

"""
    Linear <: Func
"""
@func mutable struct Linear end

forward(f::Linear, w, x, b=nothing) = b isa Nothing ? w * x : w * x .+ b

function backward(f::Linear, gy)
    w, x, b = length(f._inputs) == 3 ? f._inputs : (f._inputs..., nothing)
    gb = b isa Nothing ? nothing : sumto(gy, size(b))
    gw = matmul(gy, transpose(x))
    gx = matmul(transpose(w), gy)
    return gw, gx, gb
end

linear(w, x, b=nothing) = b isa Nothing ? Linear_F()(w, x) : Linear_F()(w, x, b)


"""
    Sigmoid <: Func
"""
@func mutable struct Sigmoid end

forward(f::Sigmoid, x) = @. tanh(x * 0.5) * 0.5 + 0.5

function backward(f::Sigmoid, gy)
    y = f.outputs[1]
    return gy .* y .* (1 - y)
end

sigmoid(x) = Sigmoid()(x)


"""
    MeanSquaredError <: Func
"""
@func mutable struct MeanSquaredError end

function forward(f::MeanSquaredError, x1, x2)
    diff = x1 .- x2
    y = sum(diff.^2) / length(diff)
    return y
end

function backward(f::MeanSquaredError, gy)
    x1, x2 = f._inputs
    diff = x1 - x2
    gx1 = gy .* diff .* (2.0 / length(diff))
    gx2 = -gx1
    return gx1, gx2
end

mean_squared_error(x1, x2) = MeanSquaredError()(x1, x2)

end