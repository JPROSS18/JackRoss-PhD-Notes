
println("---- Start of file: BifurcationDiagram.jl ----")
#=
This file is for plotting bifurcation diagrams in Julia.


First I wil learn how to plot in Julia
=#



using BifurcationKit, Plots

F(x, p) = @. p[1]*x - (x)^3 + p[2]
x0 = [-2.]
p0 = [-1., -0.00]

# Bifurcation problem (Function, initial condition, 
# parameters, parameter index)
prob = BifurcationProblem(F, x0, p0, 1;
    record_from_solution = (x,p; k...) -> x[1])

    # options for continuation
opts_br = ContinuationPar(
	# parameter interval
	p_max = 3., p_min = -1.,
	# detect bifurcations with bisection method
	# we increase the precision of the bisection
	n_inversion = 4)
# continuation paramaters (problem, method, parameters)
diagram = bifurcationdiagram(prob, PALC(), 2, opts_br)
display(plot(diagram))


#=
function saddle_node(x, p, t)
    a = p[1]
    dx =  x[1]^2 + a
    return [dx]
end



import DifferentialEquations as DE


u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = DE.ODEProblem(lorenz_f, u0, tspan, [10.0, 28.0, 8/3])
sol = DE.solve(prob)

display(plot(sol, idxs = (1, 2, 3)))

import BifurcationKit as BK


prob = BK.BifurcationProblem(lorenz_f, u0, tspan, [10.0, 28.0, 8/3])
=#
println("---- End of file ----")    
