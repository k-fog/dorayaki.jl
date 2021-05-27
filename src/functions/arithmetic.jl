import Base: +, -, *, /
import Base.Broadcast: broadcasted

export add, mul, neg, sub, pow, div

"""
    Add <: Func
"""
@func mutable struct Add end

forward(f::Add, A, B) = A .+ B

function backward(f::Add, gy)
    gx1, gx2 = gy, gy
    x_shape1, x_shape2 = size.(f.args)
    if x_shape1 != x_shape2
        gx1 = sumto(gx1, x_shape1)
        gx2 = sumto(gx2, x_shape2)
    end
    return gx1, gx2
end

add(A, B) = Add()(A, B)

+(A::Var, B::Var) = add(A, B)
+(A::Var, B) = add(A, B)
+(A, B::Var) = add(A, B)


"""
    Mul <: Func
"""
@func mutable struct Mul end

forward(f::Mul, A, B) = A .* B

function backward(f::Mul, gy)
    x1, x2 = f.args
    gx1 = gy .* x2
    gx2 = gy .* x1
    return gx1, gx2
end

mul(A, B) = Mul()(A, B)

broadcasted(::typeof(*), A::Var, B::Var) = mul(A, B)
broadcasted(::typeof(*), A::Var, B) = mul(A, B)
broadcasted(::typeof(*), A, B::Var) = mul(A, B)


"""
    Neg <: Func
"""
@func mutable struct Neg end

forward(f::Neg, x) = -x

backward(f::Neg, gy) = -gy

neg(x) = Neg()(x)

-(x::Var) = neg(x)


"""
    Sub <: Func
"""
@func mutable struct Sub end

forward(f::Sub, A, B) = A .- B

function backward(f::Sub, gy)
    gx1, gx2 = gy, -gy
    return gx1, gx2
end

sub(A, B) = Sub()(A, B)

-(A::Var, B::Var) = sub(A, B)
-(A::Var, B) = sub(A, B)
-(A, B::Var) = sub(A, B)


"""
    Pow <: Func
"""
@func mutable struct Pow{T}
    c::T
    Pow(c::T) where T = new{T}(c)
end

forward(f::Pow, x) = x.^f.c

backward(f::Pow, gy) = f.c .* f.args[1]^(f.c - 1) .* gy

pow(x, c) = Pow(c)(x)

Base.:^(x::Var, c) = pow(x, c)


"""
    Div <: Func
"""
@func mutable struct Div end

forward(f::Div, A, B) = A ./ B

function backward(f::Div, gy)
    x1, x2 = f.args
    gx1 = gy / x2
    gx2 = gy .* (-x1 / x2^2)
    return gx1, gx2
end

div(A, B) = Div()(A, B)

/(A::Var, B::Var) = div(A, B)
/(A::Var, B) = div(A, B)
/(A, B::Var) = div(A, B)