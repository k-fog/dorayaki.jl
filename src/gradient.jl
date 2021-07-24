export backward!

function backward!(top::Var; debug=false)
    sorted = topsort(top)
    isgraddefined(top) || initgrad!(top)
    for v in sorted
        f = v.creator
        f isa NullFunc && continue
        debug && println(typeof(f))
        gys = [output.value.grad for output in f._outputs]
        gxs = backward(v.creator, gys...)
        gxs isa Tuple || (gxs = (gxs,))
        for (x, gx) in zip(f._inputs, gxs)
            debug && println("\t", typeof(gx), ":", size(gx))
            isgraddefined(x) && !isnothing(gx) ? x.grad = x.grad + gx : x.grad = gx
        end
    end
    return sorted[end].grad
end