
Ergodic Theory is fundamental for the understanding of Random Dynamical Systems. Core aspects are covered in these notes. 

# Invariant Measures
An invariant measure is a measure that is preserved by a dynamical system. It is the most basic concept of Ergodic theory. 
$(X, \mathcal{F})$ is a measurable space and $f: X \to X$ a measurable space. 

**Defintion: Invariant Measure**
A probability measure $\mu : \mathcal{F} \to [0, 1]$ is called an invariant measure w.r.t the mapping $f: X \to X$ if 
$$
\mu (f^{-1}(A)) = \mu (A) \quad \forall A \in \mathcal{F} 
$$
In this course we restrict to the probability measures. 

Measures of sets $A$ do not change under the dynamics of the system. 


Why consider the pre-image and not the image? 

**Definition: Push-forward Measure**
GIven $f: X \to X$ and a mapping $f_{*}: M_1(X) \to M_1(X)$ defined by:
$$
(f_{*}\mu)(A) = \mu (f^{-1}(A)) \quad \forall A \in \mathcal{F}
$$
where $M_1(X)$ is the space of probability measures on $X$. 

Measure how much 'volume/mass' is pushed forward onto $A$ by the dynamical system w.r.t to a specific measure $\mu$. 
An invariant measure is a fixed point of the map $f_{*}$. 

$$
f_*(\mu) = \mu
$$

Moves onto consider properties of the push-forward measure w.r.t the Dirac measure. 
**Examples:**

Consider the DIrac measure $\delta_{x_0}(x)$. It can be easily deduced that:

$$
f_* \delta_{x_0} = \delta_{f(x)}
$$
It can then be seen that the $\delta_x$ is an invariant measure on fixed points ($f(x_0) = x_0$).


Additional Examples in notes

# Poincaré Recurrence Theorem

**Theorem:**
Let $(X, \mathcal{F}, \mu)$ be a probability space and consider a measurable mapping $f: X \to X$ such that $\mu$ is invariant w.r.t $f$. 

Let $A \in \mathcal{F}$ and $\mu(A) > 0$. Then for $\mu$-almost all points $x \in A$, there exists an $n \in \mathbb{N}$ such that $f^{n}(x) \in A$. 

Also, for $\mu$-almost all points $x \in A$, there are infinitely many $i \in \mathbb{N}$ for which $f^{i}(x) \in A$. 

**Proof:**
Discussed in notes. Manageable. 


**Example:**
Let $f_\alpha : \mathbb{S}^1 \to \mathbb{S}^1$   be a circle rotation, $f_\alpha(x) = xe^{2\pi i \alpha}$. It can be shown that the Lebesgue measure is invariant under $f_\alpha$.  

When $\alpha$ is rational, every orbit is periodic with the same period. Poincaré's Recurrence theorem will give us no new information. 

If $\alpha$ is irrational, let $A = B_\delta (x)$. $\mu(A) > 0$ for the Lebesgue  measure. Therefore, for all $x \in \mathbb{S}^1$ and $\forall \delta > 0$, $\exists n \in \mathbb{N}$ such that:
$$
|f^n_\alpha(x) - x| < \delta
$$
Therefore every orbit is dense in $\mathbb{S}^1$ when $\alpha$ is irrational. 

Is this a chaotic but not strange attractor? Not really an attractor?

# Theorem of Krylov-Bogolyubov 
