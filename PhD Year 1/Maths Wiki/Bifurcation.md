
A bifurcation is when a small change in a parameter of dynamical system leads to a qualitative change in system behaviour. 



**Global vs Local Bifurcations:**
When the system behaviour can be fully understood by through analysis of only a neighbourhood of the bifurcating limit set, it is called a *local bifurcation*. Otherwise it is referred to as a *global bifurcation*.

Global Bifurcations generally occur when larger invariant sets collide and a global impact. For example, the *homoclinic bifurcation* occurs when a limit cycle collides with a saddle and is destroyed. 


**Chaotic Systems:**
When a chaotic attractor is involved, bifurcations are often called a *crisis*. 

For example the *boundary crisis* occurs when a chaotic attractor intersects with its own basin of attraction. 

*Interior Crisis* is causes when the chaotic attractor collides with a unstable invariant set within the basin of attraction. 

When two chaotic attractors merge, it is called an *attractor merging crisis*. 

Nonlinear Dynamics - Datseris and Parlitz

***What is a chaotic saddle?***
A chaotic saddle is an unstable chaotic attractor. All nearby trajectory will diverge from it. 

(Due to fractal nature of such an invariant set, all trajectories will diverge from chaotic saddle almost surely (chaotic attractors are dense with unstable periodic orbits (may not be correct reason.).

Chaotic Saddles lead to *transient chaos*.


**Normal Forms:**

Bifurcations can be studied in term of their [[Normal Forms of Bifurcations |normal forms]], canonical examples which a topologically equivalent to all other bifurcations of that class.

# Local:

Degeneracy condition: 

A bifurcation point is considered non-degerate 
Consider a scalar differential equation defined by $\dot x = f(r, x)$  which has a fixed point at $(x, r) = (0, 0)$ where:
$$
\frac{\partial f}{\partial x} (0, 0)= 0, \quad \frac{\partial^2 f}{\partial x^2} (0, 0)\neq 0, \quad \frac{\partial f}{\partial r} (0, 0) \neq 0, \quad 
$$

Then this point is a non-degenerate bifurcation point. 
# Diagrams:

**Orbit:**
In practice, orbit and bifurcation diagrams often coincide. 

In an orbit diagram, allows system to evolve beyond transient for parameter and plot motion (only plots invariant sets). 

Orbit Diagrams do not distinguish between chaotic and quasiperiodic motion as both can fill the real line. However, nearby trajectories will not diverge in quasiperiodic motion. 

It is necessary to use a Poincaré section to compute the orbit diagram of a continuous system. 

Downsides:Orbit Diagrams do not capture repelling sets. 

**Bifurcation Diagram:**

Bifurcation Diagrams shows the evolution of fixed points and their stability. Numerically, solutions are found as a root finding problem.

Solutions can be found iteratively using newtons method in low dimensions or by [[Numerical Continuation]] numerical continuation in higher dimensions. 





#### Categories

Saddle-node
Pitchfork
Trans critical 



Homoclinic
Heteroclinic



[[ Hopf Bifurcation]]
**



