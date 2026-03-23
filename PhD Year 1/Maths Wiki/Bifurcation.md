
A bifurcation is when a small change in a parameter of dynamical system leads to a qualitative change in system behavoir. 

When the system behaviour can be fully understood by through analysis of only a neighbourhood of the bifurcating limit set, it is called a *local bifurcation*. Otherwise it is referred to as a *global bifurcation*.

Global Bifurcations generally occur when larger invariant sets collide and a global impact. For example, the *homoclinic bifurcation* occurs when a limit cycle collides with a saddle and is destroyed. 

When a chaotic attractor is involved, bifurcations are often called a *crisis*. For example the *boundary crisis* occurs when a choatic attractor intersects with its own basin of attraction. 

What is a choatic saddle?
A chaotic saddle is an unstable chaotic attractor. All nearby trajectory will diverge from it. 

(Due to fractal nature of such an invariant set,all trajectories will diverge from chaotic saddle almost surely (choatic attractors are dense with unstable periodic orbits (may not be correct reason.).

Chaotic Saddles lead to *transient choas*.

*Interior Crisis* is causes when the chaotic attractor collides with a unstable invariant set within the basin of attraction. 

When two chaotic attractors merge, it is called an *attractor merging crisis*. 

# Diagrams:

**Orbit:**
In practice, orbit and bifurcation diagrams often coincide. 

In an orbit diagram, allows system to evolve beyond transient for parameter and plot motion (only plots invariant sets). 

Orbit Diagrams do not distinguish between chaotic and quasiperiodic motion as both can fill the real line. However, nearby trajectories will not diverge in quasiperiodic motion. 

It is necessary to use a Poincaré section to compute the orbit diagram of a continuous system. 

Downsides:Orbit Diagrams do not campture repelling sets. 

**Bifurcation Diagram:**

Bifurcation Diagrams shows the evolution of fixed points and their stability. Numerically, solutions are found as a root finding problem.

Solutions can be found iteravely using newtons method in low dimensions or by [[Numerical Continuation]] numerical continuation in higher dimensions. 


# Local vs Global Bifurcation


#### Categories

Saddle-node
Pitchfork
Trans critical 



Homoclinic
Heteroclinic



## Hopf Bifurcation 

$$
\begin{aligned}
\dot r = r (\rho + \alpha r^2) \\
\dot \theta = \omega + \beta r^2
\end{aligned}
$$

$\dot r = 0$ when $r = 0, \sqrt{-\frac{\rho}{\alpha}}$

**In Cartesian co-ords:** 

$$
\begin{align}
\dot x = \rho x - \omega y + (\alpha x - \beta y)(x^2 + y^2) \\
\dot y = \omega x + \rho y + (\beta x + \alpha y)(x^2 + y^2)
\end{align}
$$

Which is linearised as:
$$
\begin{align}
\dot x = \rho x - \omega y \\
\dot y = \omega x + \rho y 
\end{align}
$$
Jacobian:


$$ J = \begin{pmatrix} \rho & -1 \\ 1 & \rho \end{pmatrix} $$
Eigenvalues:
$$
\lambda_{1,2} = \rho \pm \omega i
$$


The hopf bifurcation can be divided into supercritical ($\alpha < 0$ stable limit cycle) and subcritical ($\alpha > 0$ ). Similar to the pitchfork bifurcation 

**Subcritical ($\alpha$ >0):**



**Supercritical ($\alpha < 0$):**



