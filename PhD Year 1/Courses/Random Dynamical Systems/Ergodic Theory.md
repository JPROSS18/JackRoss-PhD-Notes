
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

**Theorem:**
Let $X$ be a compact metric space and $f: X \to X$ be a continuous map. Then there exists a probability measure $\mu: B(X). \to [0, 1]$ that is invariant with respect to $f$.

Compactness is essential. 

**Proof:**
Largely skipped (should return to)
$$
\mu_n = \frac{1}{n}\sum_{i =0}^{n-1} f_{*}^{i}\mu_1
$$
Show weak convergence to an invariant measure. 

# Ergodic Invariant Measures
Want a stronger notion of invariant measure with a nicer properties. 

Want the $\mu(A)$ to correspond to the proportion of the asymptotic of a trajectory which lies within a set $A$.
$$
\lim_{n \to \infty} \mathbb{1}_{A}(f^{i}(x)) = \mu(A)  
$$
for $\mu-$almost all $x \in X$. 

This is not the case, when an invariant set of $X$, does not have either measure $\{0, 1\}$ (as asymptotic dynamics take place entirely within invariant sets.)

**Defintion 2.18: (Ergodic Invariant Measure)**
$\mu: \mathcal{F} \to [0, 1]$ is called ergodic if any $A \in \mathcal{F}$ with $f^{-1}(A) = (A)$ has zero or full meusure. 

It is ergodic if it invariant and if invariant sets have full or zero measure. 

**Example: Ergodicity of Dirac Measures**

The Dirac measures of the form 
$$
\delta_p = \frac{1}{n}\sum_{i = 1}^{n} \delta_{a_i}
$$
for a period orbit $\{ a_1, .... , a_n \}$ are ergodic. 



A function $g: X \to \mathbb{R}$ is called *invariant* with respect to $f$ (or $f-invariant$) if $g \circ f = g$ almost everywhere. 

Function $g$ is invariant w.r.t the flow. Equivalently $g$ is a fixed point of the [[Koopman Operator]],  $\mathcal{K}g = g$.

**Proposition:** An invariant probability measure $\mu$ is ergodic if and only if every $f$-invariant function is constant almost everywhere. 

**Proof:**
Sketch

If $g \circ f = g$. 

Then you can show the sets 
$$
A_c = \{ x \in X \  | \ g(x) > c \}
$$
are invariant. 

If $\mu$ is ergodic then each $A_c$ has measure $\{ 0, 1 \}$. From this is can be shown all measure is concentrated at one $g(x) = c$, or that $g$ is constant a.e.

For the other direction, if $\mu$ is not ergodic. Then there exists an invariant set $A$ such that $\mu(A) \in (0, 1)$. 

Then $\mathbb{1}_A$ is a $f$-invariant funciton that is non constant a.e. 


**Note:**
The final theorem doesn't hold just in a probability space. It holds for all invariant measures. 

The definition of ergodic measure for a general measure is modified to be 
$$
\mu(A) = 0 \ \text{or} \ \mu(A^c) = 0
$$
for all invariant sets A. 

 # Birkhoff's Ergodic Theorem
Time averages are space averages !!!!!



