
Theory of RDS is in its early stages. Most results are from Ludwig Arnould and Bremen group in the 80s and 90s. 

Noise is an independent dynamical system. Statisical properties of the noise can be studied through *ergodic theory*. 

**Example: Ulam-von Neumann map**
$$
x_{n+1} = x_n^2 -2
$$
This is chaotic map (no requirement for continuouty in discrete dynamical systems. )
For almost all initial conditions $x_0 \in [-2, 2]$.

$$
\lim_{n \to \infty} \mathbb{1}_{[a,b]}(f^{i}(x_0)) = \int^b_a \frac{1}{\pi \sqrt{4 - t^2}} dt
$$
This represents the proportion of time spent by the orbit in the interval $[a, b]$. This is a special case of Birkhoff's ergodic theorem. 

$$
\frac{1}{n} \ln \left| \frac{df^n(x_0)}{dx} \right| = \frac{1}{n} \sum_{i =0}^{n-1} \ln \left| f'(f^i(x_0)) \right| 
$$
Taking $n \to \infty$, we get 
$$
\int^2_{-2} \frac{\ln |f'(t)| }{\pi \sqrt{4 - t^2}} dt = ln(2) >0
$$
for almost all $x_0 \in [-2, 2]$. This indicates positive Lyapunov exponent.

**Example: Circle Map with additive noise**

$$
f_{\alpha}(x) = x - a\sin(2\pi x) + \alpha b \quad mod \ 1
$$

Where $alpha ~ UNIF(-1, 1)$. 
The long term dynamics has bounded support when $b < a$. However, is supported on the whole interval when $b > a$. Diagram in notes. 

This course will later discuss sensitive dependence on initial conditions in RDS and how attraction will domoniate in an RDS, when it will not necessarily do so in its determinisitc counterpart.

