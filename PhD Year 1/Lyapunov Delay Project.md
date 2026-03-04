
# Autonomous Case
Consider a saddle node normal form: 
$$
\dot x = f(x) = x^2 - a \tag{1.1}
$$
$(1)$ has 2 equilibria for $a > 0$ and undergoes a saddle-node bifurcation at $a = 0$. 

Consider the stable equilibrium at $x_e = - \sqrt a$. 

Expanding $f(x)$ close to the equilibrium $x_e$:
$$
\begin{aligned}
f(x) \approx f(x_e) + \frac{\partial f}{\partial x}(x_e) (x - x_e) + O((x-x_e)^2) \\
= -2\sqrt a \ (x+\sqrt a) + O((x+\sqrt a)^2)
\end{aligned}
$$
From this we can say that there exist sufficiently small neighbourhood of $x_e$,  $B_\varepsilon(x_e) = \{  x \ | \ \ ||x_e - x|| < \varepsilon \}$. Where:
$$
f(x) = -2\sqrt a \ (x+\sqrt a) \tag{1.2}
$$

Consider the evolution of small perturbation over time $y(t)$, $y(0) = y_0, \ 0 < y_0 < \varepsilon$ to the constant solution $\mu(t), \ \mu(0) = x_e$. Which can be expresses as the initial value problem: 
$$
\dot y = \frac{\partial f}{\partial x} (\mu(t)) \cdot y, \quad y(0) = y_0 \tag{1.3}
$$
As $y_0 = B_\varepsilon(x_e)$ and $\mu(t) \equiv x_e = -\sqrt a$, from $(2)$ we get 
$$ 
\frac{\partial f}{\partial x} (\mu(t)) = -2\sqrt a
$$
Then solving $(3)$ we get: 
$$
y(t) = y_0 e^{-2 t \sqrt a}
$$
Therefore, the local exponential growth rate of pertubation to $x_e$ is $- 2\sqrt a$ < 0. As this is negative, we will always remain within $B_\varepsilon (x_e)$ for all $t > 0$. Therefore we can calculate the Lyapunov exponent by taking: 

$$
\lambda_a = \lim_{t \to \infty} \frac{1}{t} \ln(y(t)) = \lim_{t \to \infty} \left( -2\sqrt a + \frac{\ln y_0}{t} \right) = -2 \sqrt a \tag{1.4}
$$
This gives us the true Lyapunov Exponent close to the equilibrium of the autonomous system $(1)$. 



# Non-autonomous Case:

We modify equation $(1)$ to be non-autonomous by adding a linear parameter drift: 
$$
\dot x = f(t, x) = x^2 - a(t), \tag{2.1} 
$$
Consider a initial condition $(t_0, x_0)$, where $a(t_0) = a_0 > 0$ and $x_0 = -\sqrt a_0$. 

Expanding around $(t_0, x_0)$: 
$$
\begin{aligned}

f(t, x) \approx f(t_0, x_0) + \frac{\partial f}{\partial x}(t_0, x_0) (x - x_0) +  \frac{\partial f}{\partial t}(t_0, x_0) (t - t_0)  + O((x-x_0)^2, (t-t_0)^2) \\
= - 2\sqrt a_0 (x + \sqrt a_0) - \frac{da}{dt}(t_0)(t -t_0) 

\end{aligned} \tag{2.2}
$$

## Linear Rate:
Let $a(t)$ evolve with a constant rate:
$$
\quad \frac{da}{dt} = -r_0 \in \mathbb{R}
$$

Arbitrarily close to $(t_0, x_0)$, we get 

$$
\frac{\partial f}{\partial x}(t, x) = - 2\sqrt a_0, \quad \frac{\partial f}{\partial t}(t, x) = - r_0 \tag{2.3}
$$


The growth rate of perturbation in x still modelled by $(1.3)$: 
$$
\dot y = \frac{\partial f}{\partial x} (\mu(t)) \cdot y, \quad y(0) = y_0 
$$
As $a(t)$ does not depend upon x, $\frac{\partial f}{\partial x}$ is unchanged from the autonomous case. Therefore we get:

$$
y(t) = y_0 e^{-2 t \sqrt a_0} \tag{2.4}
$$
when $t \in B_\varepsilon(t_0)$. 

Restricting to a neighbourhood of $t_0$ means we cannot calculate the lyapunov exponent as in $(1.4)$. Instead we call $(2.4$) the local exponential growth rate. 

If we compute the Finite time Lyapunov Exponent on the interval $(t_0, T)$. The FTLE will approximate the average true Lyapunov exponent on the interval (this is a jump). This value can be deduced to be:

$$
\eta (t_0, T) = \frac{ a(T)^{3/2} - a(t_0)^{3/2}}{(3/4)r_0(T - t_0)} \tag{2.5}
$$

The difference between:

$$
\eta(t_1, T) - \eta (t_0, T) = \frac{4}{3r_0} \left[ \left( \frac{1}{T -t_1} - \frac{1}{T -t_0} \right) a(T)^{3/2}  - \frac{a(t_1)^{3/2}}{T - t_1} + \frac{a(t_0)^{3/2}}{T - t_0}\right ]
$$
We can computer this quantity without knowledge of $a(t)$. However it does require:

- Explicit computation of the integral: 
$$
\int^{T}_{t_0} \lambda(a(s)) ds
$$
- Knowledge of $\frac{\partial f}{\partial x}(t, x)$ to compute $\eta(t, T)$. 


Where $\lambda(a)$ is the Lyapunov exponent dependent upon the bifurcation parameter. 

## General Case:
Let $f: \mathbb{R^d} \to \mathbb{R^d}$ and $r: \mathbb{R} \to \mathbb{R}$. 
$$
\dot x = f(x(t), r(t)) \tag{3.1}
$$
be a non-autonomous ode and for a fixed $s$, let the corresponding autonomous system
$$
\dot x = f(x(t), r(s)) \tag{3.2}
$$
be the frozen system at $r(s)$

As $(3.2)$ is an autonomous system, we can the Lyapunov exponents from the variational equation: 
$$
\dot y = \frac{\partial f}{\partial x} (\mu(t), r(s)) \cdot y, \quad y(0) = y_0
$$
where $\mu(t)$ is a typical solution. 

The Lyapunov exponents of the frozen system $(3.2)$ are then defined by:
$$
\lambda (s) = \lim_{t \to \infty} \frac{1}{t} \ \ln(y(t \ |\  s)) \tag{3.3}
$$
when the limit exists. 

How do the Lyapunov exponents of the frozen system relate to the finite-time Lyapunov exponents computed on the non-autonomous system ? 

If the a Finite-time Lyapunov Exponent (FTLE) is computed on an interval $[T - L, T]$ of the non-autonomous system $(3.1)$. 

*This is an assumption - need to verify*
It take the average of a changing exponential growth rate, and therefore should approximate:

$$
\eta = \frac{1}{L}\int^{T}_{T-L} \lambda(s) ds
$$

where $\lambda(s)$ is the Lyapunov exponent of a the frozen at $r(s)$. 

For a fixed $T$, we can deduce: 

$$
\frac{d}{dL} (L\eta)= \lambda(T -L)  
$$
To compute $\frac{d}{dL} (L\eta)$:
- $\frac{\partial f}{\partial x}$ to compute the FTLE $\eta$. 

If $f = f_\theta$, this approach should allow use to recover $\lambda(s)$ without extrapolating to unseen $\dot r$. 



Questions: 
- Is the assumption that the FTLE will approach the average of $\lambda(s)$ justified? 
- 

Should be true when $\dot \lambda$ is low enough for $\mu(t)$ to sample the whole attractor for $[\lambda(r(t)) - \varepsilon, \lambda(r(t)) + \varepsilon]$. 

Should be true at an equilibrium (saddle node). 

True for a periodic solution 

- *Under what conditions does a non-autonomous system have a unique* $\lambda(s)$*?*. 




# Linear Parameter Drift:

We modify equation $(1)$ to be non-autonomous by adding a linear parameter drift: 
$$
\dot x = f(t, x) = x^2 - a(t), \tag{2.1} \quad \frac{da}{dt} = r_0 \in \mathbb{R}
$$
Consider a initial condition $(t_0, x_0)$, where $a(t_0) = a_0 > 0$ and $x_0 = -\sqrt a_0$. 




## Including t. 
Expanding $f(t, x)$ about $(t_0, x_0)$.

$$
\begin{aligned}

f(t, x) \approx f(t_0, x_0) + \frac{\partial f}{\partial x}(t_0, x_0) (x - x_0) +  \frac{\partial f}{\partial t}(t_0, x_0) (t - t_0)  + O((x-x_0)^2, (t-t_0)^2) \\
= - 2\sqrt a_0 (x + \sqrt a_0) - r_0(t  -t_0) 

\end{aligned} \tag{2.2}
$$


Now consider the exponential growth rate of a perturbation to $(t_0, x_0)$ and :

$$
\dot y = \left( \frac{\partial f}{\partial x} (t, \mu(t)) + \frac{\partial f}{\partial t} (t, \mu(t)) \right) \cdot y, \quad y(0) = y_0  \\ \tag{2.3}
$$
For a initial perturbation  $y_0$ sufficiciently small such that $2.2$ holds, we get: 

$$
\dot y = \left( \frac{\partial f}{\partial x} (t, \mu(t)) + \frac{\partial f}{\partial t} (t, \mu(t)) \right) \cdot y, \quad y(0) = y_0  \\ \tag{2.3}
$$


