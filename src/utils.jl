export numericalgrad, gradcheck

function allclose(a, b; rtol=1e-4, atol=1e-5)
    return all(@. abs(a - b) <= (atol + rtol * abs(b)))
end

function numericalgrad(f, x, args...; eps=1e-4)
    x = x isa Var ? x.data : x
    grad = zeros(size(x))
    for idx in 1:length(x)
        tmp = x[idx]
        x[idx] = tmp + eps
        fxh1 = f(x, args...)
        x[idx] = tmp - eps
        fxh2 = f(x, args...)
        grad[idx] = sum(fxh1 .- fxh2) / (2 * eps)
        x[idx] = tmp
    end
    return grad
end

function gradcheck(f, x, args...; rtol=1e-4, atol=1e-5)
    x = asvar(x)
    num_grad = numericalgrad(f, x, args...)
    y = f(x, args...)
    gradient!(y)
    bp_grad = x.grad.data
    size(num_grad) == size(bp_grad) || return false
    allclose(num_grad, bp_grad) || return false
    return true
end

function _sumto(x, shape)
    lead = ndims(x) - length(shape)
    lead_dim = Tuple(1:lead)
    dim = ()
    for i in 1:length(shape)
        if shape[i] == 1 && size(x, i) > 1
            dim = tuple(dim..., i + lead)
        end
    end
    y = sum(x, dims=(lead_dim..., dim...))
    lead > 0 && (y = dropdims(y, dims=lead_dim))
    return y
end