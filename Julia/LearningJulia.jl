
#i = 0

#=

Comment block
This is Julia Test file
=#
i = i + 1
println("---- Start of file: $i ----")


A = 1:1:10
println("size(A) = ", size(A))
A[1]
println(1:10:1)§

f = x -> x^2
println(f(2))
println(A)
for i in A
    println(f(i))
end


println("---- End of file ----")
