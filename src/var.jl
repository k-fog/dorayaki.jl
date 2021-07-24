export Var, asvar, data, gradient, initgrad!, cleargrad!, isgraddefined, addname!, name
export Param, isparam

mutable struct Var{T <: AbstractArray,F <: Func}
    data::T
    creator::F
    grad::Union{Var,Nothing}
    name::String
end

function Var(data::T, creator::F=nullfunc; name="") where {T <: AbstractArray,F <: Func}
    Var{T,F}(data, creator, nothing, name)
end
Var(data::Number, creator=nullfunc) = Var([data], creator)

asvar(x) = x isa Var ? x : Var(x)
asvar(x, creator) = Var(x, creator)

data(x::Var) = x.data

gradient(x::Var) = x.grad

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)
Base.eltype(x::Var) = eltype(x.data)

function Base.show(io::IO, v::Var)
    print(io, "Var($(v.data))")
end

function cleargrad!(x::Var)
    x.grad = nothing
end

cleargrad!(x::Var...) = cleargrad!.(x)

function initgrad!(x::Var, v=1.0)
    x.grad = Var(fill(v, size(x.data)))
end

isgraddefined(x::Var) = !isnothing(x.grad)

function setname(x::Var, name)
    x.name = name
end

getname(x::Var) = x.name