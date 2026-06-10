
Related to [[Normal Forms of Bifurcations]]


The saddle node bifurcation is the only generic co-dimension one bifurcation of a differential equation (the only being the hopf bifurcation).


# Normal Form

The standard example of the saddle node bifurcation can be written as:

$$
\frac{dx}{dt} = r + x^2 \tag{1}
$$


Any scalar differential equation of the form $\frac{dx}{dt} = f(r, x)$ which has a non-degenerate bifurcation point at $(x, r) = (0, 0)$, i.e: 
$$
f(0, 0) = 0, \quad \frac{\partial f}{\partial x} (0, 0)= 0, \quad \frac{\partial^2 f}{\partial x^2} (0, 0)\neq 0, \quad \frac{\partial f}{\partial r} (0, 0) \neq 0, \quad \tag{2}
$$
is locally topologically equivalent to Eq.$(1)$.

For higher dimensions, need to consider the centre manifold. 

it is topologically conjugate on the centre manifold. (need to read more about). 

**Why is the conjugacy not differentiable?:**
For the conjugacy to be differentiable, the eigenvalue at the equilbrium (representing stability). 

In the normal form $(1)$, the equilibria are symmetric at $x_e = \pm\sqrt{-r}$, where the eigenvalue is $\pm 2\sqrt{-r}$. 
then

Let $\dot y = g(y, r)$ experience a saddle-node bifurcation at $(0, 0)$.  If $y = h(x)$ is a smooth conjugacy between $g$ and $(1)$, then we know $g(h(x), r) = h'(x)(r + x^2)$. Then by differentiating both sides and evaluating at the equilibrium: 
$$
g'(h(\pm \sqrt r), r) = 2(\pm \sqrt {-r} )
$$
If the conjugacy is smooth, then the absolute value of the eigenvalue at the equilibrium is equal on both branches. This is not true for all saddle-node bifurcations, for example:
$$
\dot y = \mu y - e^y
$$
There, the conjugacy cannot be smooth for all saddle-node bifurcations. 

**What about the discrete case:**
Let $h$ be a smooth conjugacy between two dynamical systems:
$$
h(f(x_n)) = g(h(x_n)) = x_{n+1}
$$
then
$$
h'(f(x_n))f'(x_n) = g'(h(x_n))h'(x_n)
$$
For a fixed point: $f(x^*) = x^*,\ f'(x^*) = 1$. 
$$
h'(x^*) = h'(x^*)
$$

https://arxiv.org/pdf/2303.0115
Only differentiable conjugate if equilibria are symmetric.  

# Extended Normal Form:
[@glendinning_normal_2022] introduces a extended normal form in form:
$$
\dot y = \nu (r) - y^2 + a(r)y^3 \tag{3}
$$


Equivalence for the standard normal form is topological and not differentiable except in the hyperbolic case. 

The equivalence of this extended normal form to the standard normal form is differentiable provided the parameter-dependent coefficients $\nu$ and $a$ are chosen appropriately. 

To obtain a differentiable conjugacy between the extended and standard normal form 
$$
a(0) = \frac{2 f_{xxx}}{3f^2_{xx}} \tag{4}
$$
This is called the *Taken's coefficient* and is a measure how far the system is from a standard normal, the degree of asymmetry in branches and the proximity to a cusp bifurcation [@glendinning_normal_2022].

**Co-ordinate Transforms:**
In general, during normal form analysis - coordinate transformations are used to obtain local behaviour when parameters where zero. Co-ordinate changes are not performed away from bifurcation point and arguments about topological equivalence are used [@glendinning_normal_2022].

For example, consider a system which satisfies the non-degeneracy conditions. At the bifurcation point, it can be written as:
$$
\dot x = \frac{1}{2}f_{xx} x^2 + \frac{1}{6}f_{xxx} x^3 + O(x^4) \tag{5}
$$
Taking $y = -\frac{1}{2}f_{xx}(0, 0)x$, we get: 
$$
\dot y = -y^2 + ay^3 + O(x^4)
$$
**Note 1:** Cubic term cannot be removed by change of co-ordinates. (All of this is only considering an neighbourhood of the bifurcation point.). 

However, higher order terms, $y^k$, can be removed by the transformation $z = y + \beta y^{k - 1}$ for any $k \geq 4$.
$$
\beta = \frac{b}{k - 3}
$$
This transformation can be repeated to remove all higher order terms. This shows a differentiable conjugacy between $(3)$ and $(5)$ at the bifurcation point. 

**Theorem:**
If $f$ is $C^\infty$ and satisfies $(2)$, then on a neighbourhood of zero, the sequence of coordinate transformations defined earlier converges to a $C^\infty$ change of variables in which the equation takes the normal form:
$$
\dot{y} = -y^2 + a y^3,
\quad a = \frac{2 f_{xxx}}{3 f^2_{x x}}.
$$

https://www.numdam.org/article/AIF_1973__23_2_163_0.pdf (Discussed extensively)

## Differentiable Conjugacy of Saddle-Node Bifurcations:

Differentiable conjugacy only holds close to the bifurcation points for the saddle-node normal form. 

Why:

If two systems are smoothly conjugate, then at a stationary point $y^\ast = h(x^\ast)$, $f'(x^\ast) = g'(y^\ast)$. (They have the same linearisation.)

For a system which experiences a saddle-node bifurcation, this is always true close to the bifurcation point, as $f_{x} = g_{y} = 0$. However, far from the bifurcation point, this is a stronger condition. 



**Non-auto**
For there to be a smooth conjugacy between systems which experience a non-autonomous saddle-node, this would require: $$f'(x_e(t), r(t)) = g'(y_e(t), \mu (t)),\ \forall t$$. 
which is not true for all systems. 
If we assume no assumptions beyond the occurrence of a saddle-node bifurcation, All that is given by the is: 
 $$\lim_{t \to t_c} f'(x_e(t), r(t)) = \lim_{t \to t_c} g'(y_e(t), \mu (t)) = 0,\ \forall t$$
If we have choice of $\mu$ or $r$  can get this to hold ? 

Example of 
$$
\dot x = \mu x - e^x
$$
Both experience saddle, node bifurcations, however rates diverge significantly - core challenge to learning this. 
# Non-Autonomous Case


Fabbri et al 2003: https://www.worldscientific.com/doi/epdf/10.1142/S0219493704001103


A non-autonomous treatment of saddle-node bifurcations can be found in Li et al 2025 (https://arxiv.org/pdf/1611.09542v2). There denote the bifurcation point the *breaking point* ($t^b$) and the point when the trajectory tips as the *point of no return* ($t^*$). They prove that $t^b \leq t^*$. 






Undergrad research project on Non-auto saddle node bifurcations: https://warwick.ac.uk/fac/sci/maths/general/outreach/urss/gabriel_report.pdf?utm_source=copilot.com

Gutierrex et al 2012 - https://www.worldscientific.com/doi/epdf/10.1142/S0218127413500880 - application to canonical eltrostatic mems. 