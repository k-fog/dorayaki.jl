export Func, NullFunc, nullfunc, ParamCreator, paramcreator, @func

abstract type Func end

struct NullFunc <: Func end
const nullfunc = NullFunc()

"""
    (f::Func)(args...)
"""
function (f::Func)(args...)
    args = Tuple(asvar(x) for x in args)
    xs = [x.data for x in args]
    ys = forward(f, xs...)
    ys isa Tuple || (ys = (ys,))
    outputs = enable_backprop[] ? [asvar(y, f) for y in ys] : [asvar(y) for y in ys]
    f._inputs = args
    f._outputs = Tuple(WeakRef(output) for output in outputs)
    return length(outputs) > 1 ? outputs : outputs[1]
end


function Base.show(io::IO, v::Func)
    print(io, typeof(v))
end


function _addfields(obj)
    @assert obj.head == :struct "@func is for struct type."
    if obj.args[2] isa Symbol
        name = obj.args[2]
        obj.args[2] = :($name <: Func)
    elseif Meta.isexpr(obj.args[2], :curly)
        name = obj.args[2].args[1]
        header = obj.args[2]
        if obj.args[2].args[end] != :Func obj.args[2] = :($header <: Func) end
    elseif Meta.isexpr(obj.args[2], :<:)
        name = obj.args[2].args[1]
    else
        @error "unexpected args."
    end
    definedfield = obj.args[3]
    obj.args[3] = quote
        $definedfield
        _inputs::Tuple
        _outputs::Tuple
    end
    if length(definedfield.args) == 1
        push!(obj.args[3].args, :($name() = new()))
    end
    return obj
end


"""
    @func
"""
macro func(obj)
    return esc(_addfields(obj))
end
