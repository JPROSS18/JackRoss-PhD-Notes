# Definition
- Let $T > 0$ and let $d_\lambda, d_x \in \mathbb{N}$. 
- Let  $\lambda: [0, T] \rightarrow \mathbb{R}^{d_x}$ be a continuous function of bounded variation.  
- Let $f: \mathbb{R}^{d_x} \rightarrow \mathbb{R}^{d_x \times d_\lambda}$ be Lipschitz continuous.  
- Let $x_0 \in \mathbb{R}^{d_x}$.

A continuous path $x: [0, T] \rightarrow \mathbb{R}^{d_x}$ is said to solve a controlled differential equation, controlled or driven by $\lambda$, if:

$$
x(0) = x_0, \quad x(t) = x(0) + \int_0^t f(x(s))\, d\lambda(s) \quad \text{for } t \in (0, T]. \tag{1}
$$

Here $d\lambda(s)$ denotes a Riemann–Stieltjes integral, and $f(y(s))\, dx(s)$ refers to a matrix-vector multiplication.


**Note:**
If $\lambda$ has a bounded derivative, then it can be reduced to 
$$
\int_0^t f(x(s)) \, d\lambda(s) = \int_0^t f(x(s)) \frac{d\lambda}{ds}(s) \, ds \tag{2}
$$
**Note:**
CDE's are operators. It returns a function in path space which which takes $\lambda$ as an input.