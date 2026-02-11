
# Singular Value Decomposition

Method to factor a non-square matrix. $A \in \mathbb{C}^{n \times m}$

$$
A = U\Sigma V^{*}, U \in \mathbb{C}^{n \times n}, \ V \in \mathbb{C}^{m \times m}
$$
- U and V are orthogonal matrices -> Inverse is adjoint (transpose when dealing with real matrices.)
- $V^{*}$ is the conjugate transpose (transpose in real numbers)
- $\Sigma$ is diagonal: $\sigma_i = \Sigma_{ii}$
- $\sigma_1 \geq ... \geq \sigma_{rank(A)} \geq \sigma_i = 0$
- $r = rank(A)$, last $m - r$ columns of $V$ span $Ker(A)$

What does it geometrically: 
![[Drawing 2026-01-21 14.47.51.excalidraw]]

SVD generalises to non-square matrix ! 

## Reduced SVD

Reduce dimension to give a full rank system. Dimension reduction technique similar to PCA. 

$$
\begin{aligned}
1 \leq r \leq rank(A) \\
U_r \in \mathbb{R}^{M \times r} ...\\
A_r = U_r \Sigma_r V^{*}_r
\end{aligned}
$$
Best dimension r approximation of A under Frobenius Norm 
Explicit computation error is also known 
$$
\begin{aligned}
A_r = \arg \min ||A - B|| \ s.t.  rank(B) = r\\
|| A - A_r||_F = \sqrt{\sigma_{r+1} + .... }  
\end{aligned}
$$
# Proper Orthogonal Decomposition

An application of SVD to time-series, originating from fluid dynamics. 

Consider a function in a basis expansion. 
$$
f(x, t) = \sum^{\infty}_{k = 1} a_k(t)\phi_k(x)
$$
If $\phi_k$ are sins and cosine - Fourier series.  If they are $c_kx^k$, then its a Taylor series.

Approximate using a finite number of functions - error bound exist.
$$
f(x, t) \approx \sum^{K}_{k = 1} a_k(t)\phi_k(x)
$$
For $K \geq 1$. What is the best basis $\{ \phi_k (x) \}$. 

How does POD work ? 
We have:
$$
\begin{aligned}
\{ x_1, ... , x_m \} \\
\{ t_1, ... t_n \}

\end{aligned}
$$
The matrix $F$ has components $F_{ij} = f(x_i, t_j)$.  
**Step 1:**
Take SVD of $F = U\Sigma V^{*}$. 
- U is spatial modes - $\phi_k(x_j)$
- V temporal motion - $a_k(t_j)$
- $\sigma_i$ gives weights to modes to allows to indentify which are most importortant. 


Then use rank r approximation : 
$$
\frac{\|\mathcal{F}_r\|_F^2}{\|\mathcal{F}\|_F^2} = \frac{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2}{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_{\text{rank}(\mathcal{F})}^2}
$$

[[]]