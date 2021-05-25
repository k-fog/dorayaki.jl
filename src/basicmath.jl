import Base: +, -, *, /
import Base.Broadcast: broadcasted

"""
    Add <: Func
"""
@func mutable struct Add end

forward(f::Add, A, B) = A .+ B

add(A, B) = Add()(A, B)

+(A::Var, B::Var) = add(A, B)
+(A::Var, B) = add(A, B)
+(A, B::Var) = add(A, B)


"""
    Mul <: Func
"""
@func mutable struct Mul end

forward(f::Mul, A, B) = A .* B

mul(A, B) = Mul()(A, B)

broadcasted(::typeof(*), A::Var, B::Var) = mul(A, B)
broadcasted(::typeof(*), A::Var, B) = mul(A, B)
broadcasted(::typeof(*), A, B::Var) = mul(A, B)


"""
    Neg <: Func
"""
@func mutable struct Neg end

forward(f::Neg, x) = -x

neg(x) = Neg()(x)

-(x::Var) = neg(x)


"""
    Sub <: Func
"""
@func mutable struct Sub end

forward(f::Sub, A, B) = A .- B

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

pow(x, c) = Pow(c)(x)

Base.:^(x::Var, c) = pow(x, c)


"""
    Div <: Func
"""
@func mutable struct Div end

forward(f::Div, A, B) = A ./ B

div(A, B) = Div()(A, B)

/(A::Var, B::Var) = div(A, B)
/(A::Var, B) = div(A, B)
/(A, B::Var) = div(A, B)


"""
    MatMul <: Func
"""
@func mutable struct MatMul end

forward(f::MatMul, A, B) = A * B

matmul(A, B) = MatMul()(A, B)

*(A::Var, B::Var) = matmul(A, B)
*(A::Var, B) = matmul(A, B)
*(A, B::Var) = matmul(A, B)