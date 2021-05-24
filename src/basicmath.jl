import Base: +, -, *, /
import Base.Broadcast: broadcasted

"""
    Add <: Func
"""
@func mutable struct Add end
forward(f::Add, A::Number, B::Number) = A + B
forward(f::Add, A::AbstractArray, B::AbstractArray) = A .+ B
add(A, B) = Add()(A, B)
+(A::Var{<:Number,<:Func}, B::Var{<:Number,<:Func}) = add(A, B)
+(A::Var{<:Number,<:Func}, B) = add(A, B)
+(A, B::Var{<:Number,<:Func}) = add(A, B)


"""
    Mul <: Func
"""
@func mutable struct Mul end
forward(f::Mul, A::Number, B::Number) = A * B
forward(f::Mul, A::AbstractArray, B::AbstractArray) = A .* B
mul(A, B) = Mul()(A, B)
broadcasted(::typeof(*), A::Var{<:AbstractArray,<:Func}, B::Var{<:AbstractArray,<:Func}) = mul(A, B)
*(A::Var{<:Number,<:Func}, B::Var{<:Number,<:Func}) = mul(A, B)
*(A::Var{<:Number,<:Func}, B) = mul(A, B)
*(A, B::Var{<:Number,<:Func}) = mul(A, B)

"""
    MatMul <: Func
"""
@func mutable struct MatMul end
forward(f::Mul, A::AbstractArray, B::AbstractArray) = A * B
mul(A, B) = Mul()(A, B)
*(A::Var{<:AbstractArray,<:Func}, B::Var{<:AbstractArray,<:Func}) = mul(A, B)
*(A::Var{<:AbstractArray,<:Func}, B) = mul(A, B)
*(A, B::Var{<:AbstractArray,<:Func}) = mul(A, B)