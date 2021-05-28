export Var, asvar, data, gradient, initgrad!, cleargrad!, isgraddefined, addname!, name

mutable struct Var{T <: AbstractArray,F <: Func}
    data::T
    creator::F
    grad::Union{Var,Nothing}
end

function Var(data::T, creator::F=nullfunc) where {T <: AbstractArray,F <: Func}
    Var{T,F}(data, creator, nothing)
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

function Base.show(io::IO, ::MIME"text/plain", v::Var)
    print(io, "Var($(v.data))")
end

function cleargrad!(x::Var)
    x.grad = nothing
end

function cleargrad!(x::Var...)
    for v in x
        cleargrad!(v)
    end
end

function initgrad!(x::Var, v=1.0)
    x.grad = Var(fill(v, size(x.data)))
end

isgraddefined(x::Var) = !isnothing(x.grad)

const varname = Dict{Var, String}()

function addname!(x::Var, name)
    varname[x] = name
end

getname(x::Var) = haskey(varname, x) ? varname[name] : ""