

The saddle-node bifurcation is one of the two generic codimension one bifurcations of stationary

points of differential equations (the other being a Hopf bifurcation).

There are only two generic co-dimension one bifurcation of stationary points, the hopf and the saddle node 


Important Terms 
**Generic:**

[[Saddle Node (Fold) Bifurcation | saddle node]]
[[Hopf Bifurcation]]


# Definitions: 

The following results are stated in [@glendinning_normal_2022]. 

Let $U, V \subset \mathbb{R}$ be open intervals and let 
$f : U \to \mathbb{R}$ and $g : V \to \mathbb{R}$ be $C^k$ with $k \ge 2$.


## Smooth Conjugacies:
The differential equations
$$
\dot{x} = f(x), \qquad 
\dot{y} = g(y),
$$
are said to be $C^r$-conjugate ($r \ge 1$) if there exists a $C^r$ diffeomorphism $h : U \to V$ such that
$$
g(h(x)) = h'(x)\, f(x), \qquad \text{for all } x \in U. \tag{1}
$$

Let $\varphi_t(x)$ and $\psi_t(y)$ denote the flows induced by $\dot{x} = f(x)$ and $\dot{y} = g(y)$, respectively.  
An equivalent formulation is:
$$
h(\varphi_t(x)) = \psi_t(h(x)). \tag{2}
$$




## Linearisation:

Let $x^\ast$ be a stationary point of $f$, then  $y^\ast = h(x^\ast)$ is a stationary point of $g$. By differentiating (1) and setting $x = x^\ast$, we see that the two stationary points have the same stability coefficient, i.e.
$$
f'(x^\ast) = g'(y^\ast).
$$

**Theorem: ($C^k$-linearisation theorem).**  
Suppose $f : U \to \mathbb{R}$ is $C^k$ with $k \ge 2$, and  $x^\ast \in U$ is a stationary point of $\dot{x} = f(x)$ with
$$
f'(x^\ast) = \lambda \neq 0.
$$
Then there exist neighbourhoods $U_0 \subseteq U$ of $x^\ast$ and  $V_0 \subset \mathbb{R}$ of $0$ such that the system
$$
\dot{x} = f(x) \quad \text{on } U_0
$$
is $C^k$-conjugate to
$$
\dot{y} = \lambda y \quad \text{on } V_0.
$$



## Linearisation + Conjugacy

Let $x^\ast$ be a stationary point of $f$, then  $y^\ast = h(x^\ast)$ is a stationary point of $g$. By differentiating (1) and setting $x = x^\ast$, we see that the two stationary points have the same stability coefficient, i.e.
$$
f'(x^\ast) = g'(y^\ast).
$$

**Corollary:**  
Suppose $f : U \to \mathbb{R}$ and $g : V \to \mathbb{R}$ are $C^k$ with $k \ge 2$, and the systems
$$
\dot{x} = f(x), \qquad \dot{y} = g(y),
$$
have stationary points $x^\ast \in U$ and $y^\ast \in V$ satisfying
$$
f'(x^\ast) = g'(y^\ast) \neq 0.
$$
Then there exist neighbourhoods $U_0 \subseteq U$ of $x^\ast$ and $V_0 \subseteq V$ of $y^\ast$ such that
$\dot{x} = f(x) \quad \text{on } U_0$ and $\dot{y} = g(y) \quad \text{on } V_0$ are $C^k$-conjugate.


(If two systems have the same linearisation, they are $C^k$-conjugate. As linearisation is determined by first derivate - equivalent to saying first derivatives are equal).

## Extension to Basin of Attractions:

A proof extending differentiable conjugacy to the entire basin of attraction/repulsion of fixed point can be found in [@glendinning_normal_2022]. This proof extends previous proves in the discrete time context to continuous systems. 

