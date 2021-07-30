export get_dot_graph, plot_dot_graph, numericalgrad, gradcheck

function topsort(top::Var)
    sorted = Var[]
    visited = Set{Var}()
    function visit(x)
        x in visited && return
        push!(visited, x)
        x.creator isa NullFunc && return push!(sorted, x)
        for v in x.creator._inputs
            v isa Var && visit(v)
        end
        push!(sorted, x)
    end
    visit(top)
    reverse(sorted)
end


function _dot_var(v::Var, verbose=false)
    name = getname(v)
    if verbose && isdatadefined(v)
        if name != "" name *= ": " end
        name *= string(size(v)) * " " * string(eltype(v))
    end
    return "$(objectid(v)) [label=\"$name\", color=orange, style=filled]\n"
end

function _dot_func(f::Func)
    func_name = split(string(typeof(f)), ".")[end]
    txt = "$(objectid(f)) [label=\"$(func_name)\", color=lightblue, style=filled, shape=box]\n"
    for x in f._inputs
        txt *= "$(objectid(x)) -> $(objectid(f))\n"
    end
    for y in f._outputs
        txt *= "$(objectid(f)) -> $(objectid(y.value))\n"
    end
    return txt
end

function get_dot_graph(output::Var; verbose=true)
    txt = ""
    variables = topsort(output)
    txt *= _dot_var(output, verbose)

    for v in variables
        f = v.creator
        f isa NullFunc && continue
        txt *= _dot_func(f)
        for x in f._inputs
            txt *= _dot_var(x, verbose)
        end
    end
    return "digraph g {\n" * txt * "}"
end

function plot_dot_graph(output::Var; verbose=false, file="graph.png")
    dot_graph = get_dot_graph(output, verbose=verbose)
    graph_path = tempname() * ".dot"
    open(graph_path, "w") do f
        write(f, dot_graph)
    end

    extension = split(file, ".")[end]
    cmd = `dot $graph_path -T $extension -o $file`
    run(cmd)
end


function allclose(a, b; rtol=1e-4, atol=1e-5)
    return all(@. abs(a - b) <= (atol + rtol * abs(b)))
end


function numericalgrad(f, x, args...; eps=1e-4)
    x = x isa Var ? copy(x.data) : copy(x)
    grad = zeros(Float64, size(x))
    for idx in 1:length(x)
        tmp = copy(x[idx])
        x[idx] = tmp + eps
        fxh1 = f(x, args...)
        x[idx] = tmp - eps
        fxh2 = f(x, args...)
        if fxh1 isa Var grad[idx] = sum(fxh1.data - fxh2.data) / (2 * eps)
        else grad[idx] = sum(fxh1 - fxh2) / (2 * eps) end
        x[idx] = tmp
    end
    return grad
end

function gradcheck(f, x, args...; rtol=1e-4, atol=1e-5)
    x = asvar(x)
    num_grad = numericalgrad(f, x, args...)
    y = f(x, args...)
    backward!(y)
    bp_grad = x.grad.data
    size(num_grad) == size(bp_grad) || return false
    allclose(num_grad, bp_grad) || return false
    return true
end