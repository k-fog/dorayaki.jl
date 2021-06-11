export MeanSquaredError, mean_squared_error

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
    x1, x2 = f.args
    diff = x1 - x2
    gx1 = gy .* diff .* (2.0 / length(diff))
    gx2 = -gx1
    return gx1, gx2
end

mean_squared_error(x1, x2) = MeanSquaredError()(x1, x2)
