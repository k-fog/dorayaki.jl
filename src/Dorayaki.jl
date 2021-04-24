module Dorayaki

export Variable

mutable struct Variable{T <: Number}
    data::Array{T}
    Variable(data::Float64) = new{Float64}(data)
end


end
