export gradient!

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