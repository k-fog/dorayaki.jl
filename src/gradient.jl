export gradient!

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

function addto!(x, y)
    if isnothing(x) x = y
    else x = x + y end
end

function gradient!(top::Var)
    sorted = topsort(top)
    isgraddefined(top) || initgrad!(top)
    for v in sorted
        f = v.creator
        f isa NullFunc && continue
        gys = [output.value.grad for output in f.outputs]
        gxs = backward(v.creator, gys...)
        gxs isa Tuple || (gxs = (gxs,))
        for (x, gx) in zip(f.args, gxs)
            isgraddefined(x) ? x.grad = x.grad + gx : x.grad = gx
        end
    end
    return sorted[end]
end