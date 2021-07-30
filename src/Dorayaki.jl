module Dorayaki

using CUDA

const NDArray{T,N} = Union{Array{T,N},CuArray{T,N}}

include("config.jl")
include("func.jl")
include("var.jl")
include("gradient.jl")
include("layers.jl")
include("utils.jl")

include("functions/Functions.jl")
using .Functions

include("layers/Layers.jl")

function __init__()
    use_gpu[] = CUDA.functional()
    println("GPU : ", use_gpu[])
end

end
