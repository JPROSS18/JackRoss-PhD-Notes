
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


# With Linear Parameter Drift:

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


