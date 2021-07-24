module Dorayaki

using CUDA

export NN

const NDArray{T,N} = Union{Array{T,N},CuArray{T,N}}

include("config.jl")
include("func.jl")
include("var.jl")
include("gradient.jl")
include("layers.jl")
include("utils.jl")

include("functions/arithmetic.jl")
include("functions/basicmath.jl")
include("functions/tensor.jl")
include("functions/reduction.jl")
include("functions/nn.jl")

function __init__()
    use_gpu[] = CUDA.functional()
    println("GPU : ", use_gpu[])
end

end
