
Given a discrete time dynamical system 

$$
x_{n+1} = F(x_n), \quad F: \mathcal{M} \to \mathcal{M} \subset \mathbb{R^d}
$$
Let $\phi: \mathcal{M} \to \mathbb{R}$ be observables of the dynamical system. 

The Koopman Operator is a linear operator that takes an observable function and evolves in time. 

$$
\mathcal{K}\phi \to \phi  \circ  F
$$
This represents an infinite dimensional dynamical system, which evolved observable functions in time. 

If we take two set of measurements such that $Y = F(X)$. 
$$
\begin{aligned}
X = \{ X_1, X_2, .... X_n\} \in \mathbb{R}^{m \times n} \\
Y = \{Y_1, Y_2, ....,Y_n\} \in \mathbb{R}^{m \times n}\\
\end{aligned}
$$

$$
\begin{aligned}
\Phi (X) = \{ \phi(X_1), \phi(X_2), .... \phi(X_n)\} \in \mathbb{R}^{1 \times n} \\
\Phi (Y) = \{\phi(Y_1), \phi(Y_2), ....,\phi(Y_n)\} \in \mathbb{R}^{1 \times n}\\
\end{aligned}
$$

The Koopman operators will evolve as such:
$$
\mathcal{K}(\Phi(X)) = \Phi(Y)
$$

#  EDMD
Take a dictionary of observables $\mathcal{D} = \{ \phi_1, \phi_2, .... , \phi_K \}$.

Apply to data: 
$$
\Psi(X) =
\begin{bmatrix}
\psi_1(X_1) & \psi_1(X_2) & \cdots & \psi_1(X_n) \\
\psi_2(X_1) & \psi_2(X_2) & \cdots & \psi_2(X_n) \\
\vdots      & \vdots      & \ddots & \vdots      \\
\psi_K(X_1) & \psi_K(X_2) & \cdots & \psi_K(X_n)
\end{bmatrix}
\in \mathbb{C}^{K \times n}
$$

$$
\Psi(Y) =
\begin{bmatrix}
\psi_1(Y_1) & \psi_1(Y_2) & \cdots & \psi_1(Y_n) \\
\psi_2(Y_1) & \psi_2(Y_2) & \cdots & \psi_2(Y_n) \\
\vdots      & \vdots      & \ddots & \vdots      \\
\psi_K(Y_1) & \psi_K(Y_2) & \cdots & \psi_K(Y_n)
\end{bmatrix}
\in \mathbb{C}^{K \times n}
$$

Want to solve the this minimise problem: 
$$
\arg\min_{A \in \mathbb{C}^{M \times M}}
\;\bigl\| \Psi(Y) - A\,\Psi(X) \bigr\|_F^2
$$
$A$ is an approximation of the Koopman Operator. 