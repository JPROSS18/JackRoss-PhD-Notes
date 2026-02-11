# Definition: 
Assume we have two matrices
$$
\begin{aligned}
X = \{ X_1, X_2, .... X_n\} \in \mathbb{R}^{n \times m} \\
Y = \{Y_1, Y_2, ....,Y_n\} \in \mathbb{R}^{n \times m}\\
\end{aligned}
$$ 
Where each column vector $X_i \in \mathbb{R}^m$ is a state space measurement and $Y_i \in \mathbb{R}^m$ is the corresponding measurement one time step later. 

- Want to find a matrix $A \in \mathbb{R}^{m \times m}$  such that $Y \approx AX$. 
$$
\hat A = \arg \min_A  || Y - AX ||_F 
$$
Optimal $\hat A$ is given through standard linear regression  (pseudo-inverse of X used, X is not guaranteed to be square). 


Then by proper orthogonal decomposition: 
$$
X = U \Sigma V^{*} \to X \approx V \Sigma U^{*}
$$
$\hat A$ can be used to evolve into the future every cheaply. 

As A is linear so can be decomposed to eigenvalues and eigenvectors. 

Future iterations can be cheaply explained through
$$
Z_n = \sum_{m=1}^{M} c_m \mu_m^n \psi_m
$$
where eigenvectors $\phi_m$ are the DMD modes which correspond to spatial patterns, whose behaviour can be modelling by the eigenvalues.

**Notes:**
- This method is different to local linearisation, as there is no constraint that the sample points $X_i$ are close to each other.
- Used to create low-rank approximations also 

# Behaviour 


$$

\lim_{n \to \infty} \left| \mu_m^n \right| =
\begin{cases}
0, & |\mu_m| < 1, \\
1, & |\mu_m| = 1, \\
\infty, & |\mu_m| > 1.
\end{cases}
\tag{2.7}

$$


Long-time stable evolution under the DMD matrix is goverened by the eigenvalues $\mu_m$

Furthermore:
$$


\operatorname{rank}(A = Y X^{\dagger})
   \;\le\;
   \min\!\bigl\{ \operatorname{rank}(Y),\; \operatorname{rank}(X^{\dagger}) \bigr\}.
$$




In practice if there are $\mu_m >1$, the matrix will diverge. 





# Variants:
## piDMD

Physics Informed DMD: 

In the non-linear non-linear Schrödinger equation (NLS), it known that: $\int_{-L}^{L} \lvert u(x,t) \rvert^{2} \, dx$ is constant over time. 

To incorporate this into the DMD, the constraint $|| AZ_n ||_2 = ||Z_n ||$, (A is unitary). 
This adjusts the problem to
$$
\hat A = \arg \min_{A^*A = I}  || Y - AX ||_F 
$$


This is Procrustes problem - known solutions.

All eigenvalues will lie on the unit circle. 

Accurate short term forecasts which do worse vs stable long term forecasts with more error. 

## Sliding Window DMD

DMD is good at capturing fast timescale motion but not slow-scale drift. 

N must be large to even see slow scale. 

## Time Delay 


High dimension is needed (M). To allow for multiple DMD modes.

## Extended DMD (EDMD)

