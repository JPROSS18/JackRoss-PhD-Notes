
Many methods require high (or multidimensional data). In order to apply these methods to data with insufficiently large dimension, time delay co-ordinates are needed

What if we have a single measurement of a more complex system? - not enough data to get a complete picture/not all variables can be measured. 

Takens theorem shows that delay-coordinate embeddings can be used to reconstruct the state space. 
# Takens's Theorem: 

**Takens' theorem** is the 1981 delay [embedding](https://en.wikipedia.org/wiki/Embedding "Embedding") theorem of [Floris Takens](https://en.wikipedia.org/wiki/Floris_Takens "Floris Takens").#

Take a discrete time dynamical system

$$
f(x_n) = x_{n+1}, \ f: M \to M 
$$
where $M$ is a v dimensional manifold. Assume $f$ has a strange attractor, $A \subset M$ with fractal dimension $d_A$. 

It is known from Whitney's embedding theorem that A can be embedded in a manifold in $k$ dimensional where $k > 2d_A$. 

Taking a observation function $\alpha: M \to \mathbb{R}$ (with some constraints). 
Then
$$
\varphi_T(x)
=
\bigl(
\alpha(x),\;
\alpha(f(x)),\;
\alpha(f^{2}(x)),\;
\dots,\;
\alpha(f^{k-1}(x))
\bigr)
$$

is an embedding into $\mathbb{R}^k$.

**Conditions on $\alpha$
$$\alpha \in C^2$$
Must be typical ?? 


# DMD with Time-Delay 
The usual DMD minimization problem (2.3) can be reformulated by arranging
the data into Hankel matrices:

$$

X =
\begin{bmatrix}
x_1     & x_2     & \cdots & x_{N-\tau-1} \\
x_2     & x_3     & \cdots & x_{N-\tau}   \\
\vdots  & \vdots  & \ddots & \vdots       \\
x_{\tau} & x_{\tau+1} & \cdots & x_{N-1}
\end{bmatrix},
\qquad
Y =
\begin{bmatrix}
x_2     & x_3     & \cdots & x_{N-\tau}   \\
x_3     & x_4     & \cdots & x_{N-\tau+1} \\
\vdots  & \vdots  & \ddots & \vdots       \\
x_{\tau+1} & x_{\tau+2} & \cdots & x_{N}
\end{bmatrix}
\in \mathbb{C}^{d\tau \times (N-\tau-1)}.
$$


Notice that the rows and columns of \(X\) and \(Y\) are successive snapshots of the data.

We then constrain the DMD matrix to take the block‑companion form


$$
A =
\begin{bmatrix}
0_d & I_d & 0_d & \cdots & 0_d \\
0_d & 0_d & I_d & \cdots & 0_d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0_d & 0_d & 0_d & \cdots & I_d \\
A_{\tau} & A_{\tau-1} & A_{\tau-2} & \cdots & A_0
\end{bmatrix}.
$$
Here the final row creates the next iteration while all other rows save previous measurements (to preserve time delay.)

 