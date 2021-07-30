module Functions

using CUDA
using ..Dorayaki

export forward, backward

include("arithmetic.jl")
include("basicmath.jl")
include("tensor.jl")
include("nn.jl")

end