

$$
\dot y = \frac{\partial}{\partial x}f(t, \phi(t, x_0))\cdot y
$$

The Variational Equation:
$$
\dot \Phi_t(x_0) =  \frac{\partial}{\partial x}f(t, \phi(t, x_0)) \cdot \Phi_t(x_0) \quad \Phi_0(x_0) = I
$$
Models how a system $\dot x = f$ will behave under pertubation. 
Let ${m_1(t), m_2(t), ... }$ be the eigenvalues of $\Phi_t(x_0)$. 
Then:
$$
\lambda_i = \lim_{t \to \infty} \frac{1}{t}\ln | m_i(t) |
$$
defined the $ith$ Lyapunov exponent. 




Sum of Lyapunov Exponents being negative is necessary for a system to be dissaptive. 

It is possible to reconstruct largest lyapunov exponent only from time series data and full sprectrum using machine learning. 

**Methods of Calculation:**

Standard methods are by QR decomposition: 
https://carretero.sdsu.edu/teaching/M-638/lectures/ProgTheorPhys_1990-Geist-875-93.pdf

More simple notation: 
$$
\dot x = f(x)
$$

$$
\dot Y = J(x) Y, Y(0)
$$


first proposed by 

4 V. I. Oseledec, “A multiplicative ergodic theorem, Lyapunov characteristic

numbers for dynamical systems,” Trans. Moscow Math. Soc. 19, 197–231 (1968).**

https://link.springer.com/article/10.1007/s11071-018-4544-z