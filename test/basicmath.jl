using Dorayaki
using Test

x = Var(10)
@test (x + 100).data == 110
@test (100 + x).data == 110
@test (x * 100).data == 1000
