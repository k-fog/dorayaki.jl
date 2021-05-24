export Func, nullfunc, @func

abstract type Func end

struct NullFunc <: Func end
const nullfunc = NullFunc()

function (f::Func)(args...)
    args = Tuple(asvar(x) for x in args)
    xs = [x.data for x in args]
    ys = forward(f, xs...)
    ys isa Tuple || (ys = (ys,))
    outputs = enable_backprop[] ? [asvar(y, f) for y in ys] : [asvar(y) for y in ys]
    f.args = args
    f.outputs = Tuple(WeakRef(output) for output in outputs)
    return length(outputs) > 1 ? outputs : outputs[1]
end

macro func(obj)
    @assert obj.head == :struct "@func is for struct type."
    if obj.args[2] isa Symbol
        name = obj.args[2]
        obj.args[2] = :($name <: Func)
    else
        name = obj.args[2].args[1]
    end
    definedfield = obj.args[3]
    obj.args[3] = quote
        $definedfield
        args::Tuple
        outputs::Tuple
        $name() = new()
    end
    esc(obj)
end