export Var
export data, gradient, initgrad!, isgraddefined, isparam, gradient!

mutable struct Var{T,F <: Func}
    data::T
    creator::F
    grad::Union{Var{T},Nothing}
end

function Var(data::T, creator::F=nullfunc) where {T,F <: Func}
    Var{T,F}(data, creator, nothing)
end

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
    println(io, "$(eltype(v)) Var(\ndata:$(v.data)\ngrad:$(v.grad)\n)")
end

function initgrad!(x::Var, v=1)
    x.grad = Var(fill(v, size(x.data)))
end

isgraddefined(x::Var) = !isnothing(x.grad)
    
function topsort(top::Var)
    sorted = Var[]
    visited = Set{Var}()
    function visit(x)
        x in visited && return
        push!(visited, x)
        x.creator isa NullFunc && return push!(sorted, x)
        for v in x.creator.args
            v isa Var && visit(v)
        end
        push!(sorted, x)
    end
    visit(top)
    reverse(sorted)
end

function gradient!(top::Var)
    sorted = topsort(top)
    isgraddefined(top) || initgrad!(top)
    for v in sorted
        println(v.data, "($(v.creator))")
        if !(v.creator isa NullFunc) && isnothing(v.grad)
            initgrad!(v, 0)
        end
        isnothing(v.grad) && continue
        isempty(v.creator.args) && continue
        gys = [output.value.grad for output in f.outputs]
        backward(v.creator, gys...)
    end
    sorted
end