Notes + Thoughts + Ideas

Kloeden and Rasmussen 2011 

Chapters 1 - 3

Chapter 8 on Bifurcations

Chapter 11 on discretisations 

# Autonomous Dynamical Systems

Continuous time autonomous dynamical systems can be represented in the form 
$$
\dot x = f(x) \tag{1.1}
$$
where $f: \mathbb{R}^d \to \mathbb{R}^d$ is continuous function. 

The mapping 
$$
(t, t_0, x_0) \to x(t, t_0, x_0) \in \mathbb{R}^d
$$
if the general solution to (1.1). 

**Solutions to Autonomous Dynamical Systems are translation invariant in time.** 
Therefore general solution can be reduced to $\phi : \mathbb{R} \times \mathbb{R}^d \to \mathbb{R}^d$.

**Dynamical System:**
Let $\mathbb{T} = \mathbb{R}$ or $\mathbb{T} = \mathbb{Z}$ and $X$ a metric space. 
A *dynamical system* is a continuous function $\phi : \mathbb{T} \times X \to X$
1. *Initial Value Condition:* $\phi(0, x_0) = x_0 \quad \forall x \in X$ 
2. *Group Property:* $\phi(s+t, x_0) = \phi(s, \phi(t, x_0)), \quad \forall s, t \in \mathbb{T}$ and $x_0 \in X$. 

The family of mappings $\{ \phi(t, \cdot): t \in \mathbb{T} \}$ is a group. 


*Note: this definition required evolution both forward and backward in time, which is not always possible. (Non invertable $f$ in a discrete system.). This details required a weaker definition. 

**Semi-dynamical System:**
Let $\mathbb{T} = \mathbb{R}$ or $\mathbb{T} = \mathbb{Z}$ and $X$ a metric space. 
A *dynamical system* is a continuous function $\phi : \mathbb{T}_{0}^{+} \times X \to X$
1. *Initial Value condition:* $\phi(0, x_0) = x_0 \quad \forall x \in X$ 
2. *Semi-group property:* $\phi(s+t, x_0) = \phi(s, \phi(t, x_0)), \quad \forall s, t \in \mathbb{T}^{+}_{0}$ and $x_0 \in X$. 

The family of mappings $\{ \phi(t, \cdot): t \in \mathbb{T} \}$ is a semi-group.

When $f$ is a homeomorphism, then $\phi$ is a dynamical system. 

## Local Asymptotic Behaviour

## **Invariance:**
For a semi-dynamical system defined by $\phi: \mathbb{T}^{+}_0$, a subset $M \subset X$ is called invariant under $\phi$ if 
$$
\phi (t, M) = M, \quad \forall t \in \mathbb{T}_0^{+}
$$

Simple invariant sets are given by equilbria and period solutions. Can also be more complex - Lorenz 


Positively invariant if $\phi(t, M) \subset M$. Negatively Invariant if $M \subset \phi(t, M)$ 
- Ball around stable fixed point is positively invariant 
- Small subset in the Lorenz attractor is negatively invariant. 


$\omega - \text{and} \ \alpha -\text{limit sets:}$   

Notion of an attractor in forward and reverse time. 

The $\omega \text{ limit set}$ of a point $x_0 \in X$ is 
$$
\omega(x_0) := \{\, x \in X \mid \exists (t_j)_{j\in\mathbb{N}} \subset T_0^{+}
\text{ with } t_j \to \infty \text{ and } \varphi(t_j,x_0) \to x \,\}.
$$
The alpha limit set is defined similarly for reverse time. 

**Exercise 1.10:** Completed apart from final part (cts time implies connect $\omega$-limit sets)

**Exercise 1.11:** Not Completed

## Lyapunov Stability and Attractivity:

What happens close to invariant sets. 

**Lyapunov Stability:**
An invariant subset $M$ is said to be Lyapunov stable for a semi-dynamical system $\phi$ on $X$ if:

$$
\begin{aligned}
\forall \varepsilon > 0 \quad \exists \delta = \delta(\varepsilon) > 0 \ \ s.t \\
dist(\phi(t, \xi), M) < \varepsilon \ \text{for all} \ t \in \mathbb{T}^{+} \ \text{whenever} \ dist(\xi, M) < \delta
\end{aligned}
$$
When initial point $\xi$, is sufficiently close. Trajectory will stay within $\varepsilon$ forever. 

If $M$ Lyapunov stable, then $\overline M$ is also Lyapunov stable. 

**Lyapunov Attractivity:**
An invariant subset $M$ of a metric space $(X, d)$ is *lyapunov attractive* for a semi-dynamical system $\phi$ on $X$ if $\exists \delta > 0$ such that: 
$$
\lim_{t \to \infty} dist(\phi(t, x), M) = 0
$$
for all $x \in X$ with $dist(x, M) < \delta$. 


Stable vs Attractive: 
Independent concepts. 
If trajectory start sufficiently close -> it will stay close
If a trajectory starts sufficiently  close -> it will approach the set. 

How can something be attractive but not stable??? 