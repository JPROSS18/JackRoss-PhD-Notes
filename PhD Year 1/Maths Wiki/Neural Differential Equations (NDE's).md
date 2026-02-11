Neural Differential Equations

>[!quote] Neural differential equations (NDEs) demonstrate that neural networks and differential equation are two sides of the same coin — (Kidger, 2022)  


A neural differential equation is a differential equation which uses a neural network to parameterise the vector field. The simplest example if a **Neural Ordinary Differential Equation**, introduced in [[chenNeuralOrdinaryDifferential2018 |  (Chen et al., 2018) ]]



A Neural ODE can be described by: 

$$
y(0) = y_0, \; \; \frac{dy}{dx}(t) = f_\theta(t, y(t))
$$
where $f_\theta$ is a neural network. 

In contrast of other machine learning methods, which often try to fit a function to a time series $\{ x_1, ... , x_n\}$. NDE's assume the time series sampled from a latent continuous trajectory $z(t)$ 

$$
z: [0, T] \to \mathbb{R}^{d_z}, \quad z(0) = h(x, \theta_h)
$$
where $h: \mathbb{R}^{d_x} \to \mathbb{R}^{d_z}$.

NDE's attempt to learn a corresponding differential equation. 

Differential Equations have been the primary tool of mathematical modelling since Newton, with that expected to continue. NDE's attempt to incorporate universal approximation properties which neural network which drive modern machine learning into this centuries old framework. 

The continuous nature of NODE makes them ideal for irregularly spaced data.  
# Neural ODE's

First introduced in (Chen et al., 2018).

Given time series data $\{ x_1, ... , x_n\}$, *Neural Ordinary Differential Equations* attempt to learn the continuous system, $z(t)$, which generated the time series: 
$$
\frac{d}{dt}z = f_{\theta}(z(t), t)
$$
If this ODE was to be discretised by the a Euler scheme, we would get 

$$
z_{k+1} = z_k + \Delta_t f_\theta(z_k)
$$
which for $\Delta_t = 1$, reduces to a [[Residual Neural Network]]. 



Usually, $f_{\theta}$ is a feedforward neural network and it treated as a black-box approximator for the vector field $\frac{dz}{dt}$.  

Then $f_\theta$  be integrated numerically with standard ODE solvers, to get trajectories from the system.  

Standard Neural ODE's use [[Multi-Layer Perceptron (MLP)| | Multi-Layer Perceptrons ]] to get $f_{\theta}$. As all weights are trainable. Therefore, the number of parameters can be vary large (in thousands to possibly millions, even a small network will have ~1000 parameters). Such high dimensionality makes interpretability of results very difficult. 

However, what NODE's lack in interpretability, they make up for in flexibility to interpolate and generalise to unseen regimes. 

### Issues 

#### Error Propagation 
May propagate errors when trained using standard methods. 

$$
x^{true}_0 \to x_1 \to x_2 \to x_3
$$
Then we must compute loss (MSE example shown):
$$
\mathcal{L}(x, x^{true}) =\frac{1}{N} \sum^N_{i=1} || x_i - x^{true}_i||^2
$$
When N is large, $x_i$ are very far from any true values. This could allow errors to propagate (numerical solver error influencing loss + product of large number of terms need during backpropagation). 

#### Functions which cannot be learnt by standard NODE's

- Standard NODE's are not universal approximators. 
- 

Dupoint et al (2019) prove that the evolution operator $\phi$ corresponding to dynamical learned by a standard neural ode is a homeomorphism. - need to think about this a lot more ? **$\phi_T(x)$ has the same topology as $x$?**  

**Therefore the input space will be topologically equivalent to the output space.** This means the evolution of neural ode can only continuously dormef the input space. 

However, It is possible for a neural ode to well approximate functions which lie outside this class. (It will overall poor solutions). Examples can be found at https://github.com/EmilienDupont/augmented-neural-odes. For example, they also show Resnet can learn functions neural odes cannot. This can be explained as the high numerical error allow trajectories to cross. 



#### Increasing Computational Complexity

As training progresses, the flow becomes increasingly complex ? why? (needs more time to add more complexity to the network ?)

ODE becomes more computationally expensive to solve after longer computations. 


Weight decay Grathwohl et al ?
## Augmented Neural ODE's:

There are constraints on the approximation properties of the Neural ODE's in their simplest form. 

The latent space on which the ode evolves is augmented from $\mathbb{R}^d$ to $\mathbb{R}^{d + p}$. This will allow trajectories to avoid intersecting each other.  

> [!quote] However, autonomous ODE flows may limit expressivity by constraining trajectories to remain diffeomorphic transformations of initial states, thus prohibiting intersecting paths in latent space. - This is interesting I need to read more about it 



[[dupontAugmentedNeuralODEs2019| Dupont et al. (2019)]] introduced Augmented Neural ODE's to overcome limitations: 
$$
\frac{d}{dt}
\begin{bmatrix}
z(t) \\
a(t)
\end{bmatrix}
= f\bigl(\begin{bmatrix}
z(t) \\
a(t)
\end{bmatrix},

\space t; \space\theta_f\bigr)
 \quad 
 \begin{bmatrix}
z(0) \\ 
a(0)
\end{bmatrix}
= 
\begin{bmatrix}
x \\
0
\end{bmatrix}
$$

Where $a(t)$ is an auxiliary variable. 

This approach should make the learned $f$ smoother. Further details in review of paper. 

ANODE generalise much better and lower computation cost (as less complicated flows are learned). 

- ANODE outperforms NODE even when total number of parameters are equivalent. 

## Training 
Training regimes can be divided into discretise-then-optimise and optimise-then-discretise. 
### Discretise-then-optimise:

Simply [[Backpropagation| backpropagate]] through through operations of the solver. 

$$
\begin{aligned}
\nabla_\theta \mathcal{L} = \frac{1}{N}\sum^N_{i = 1} || x_i + x^{true}_i||^2 \\
 x_i = x(t_i) = x(0) + \int^{t_i}_{t_0} f_\theta(x(s), s) ds 
\end{aligned}
$$
For a Euler Scheme:
$$
\begin{aligned}
x_{i+1} = x_{i} + f_\theta(x_i, t_i),\quad \text{let} \space f_i(x) = f_\theta (x, t_i).  \\ 
x_{i+1} = x_i + f_i \circ .... \circ f_0(x_0, )\\
x_{n+1} = x_0 + f_0(x_0) + \sum^n_{k = 1} f_k \circ .... \circ f_0(x_0 )
\end{aligned}
$$
Then to take gradient with MSE:
$$
\begin{aligned}
\nabla_\theta (x_i - x^{true}_i )^2 = 2(x_i - x^{true}_i)\nabla_\theta x_i \\

\nabla_\theta x_i = \nabla_\theta (x_0 + f_0(x_0) + \sum^{i-1}_{k = 1} f_i \circ .... \circ f_0(x_0 )) \\
= \nabla_\theta f_0 + 
\end{aligned}
$$



### Optimise-then-discretise ('the adjoint method')



Let $y_0 \in \mathbb{R}^d$ and $\theta \in \mathbb{R}^m$. Let  $f_\theta : [0, T] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$  be continuous in $t$, uniformly Lipschitz in $y$, and continuously differentiable in $y$. Let  $y : [0, T] \rightarrow \mathbb{R}^d$  be the unique solution to:

$$
y(0) = y_0, \quad \frac{dy}{dt}(t) = f_\theta(t, y(t)).
$$


**How do you optimise w.r.t $\theta$ ?**

Let $L = L(y(T))$ be some (for simplicity scalar) function of the terminal value $y(T)$. (Loss function on output)

Then:

$$
\frac{dL}{dy(T)} = a_y(T), \quad \frac{dL}{d\theta} = a_\theta(0),
$$

where $a_y : [0, T] \rightarrow \mathbb{R}^d$ and $a_\theta : [0, T] \rightarrow \mathbb{R}^m$ solve the system of differential equations:

$$
a_y(T) = \frac{dL}{dy(T)}, \quad \frac{da_y}{dt}(t) = -a_y(t)^\top \frac{\partial f_\theta}{\partial y}(t, y(t));
$$

$$
a_\theta(T) = 0, \quad \frac{da_\theta}{dt}(t) = -a_y(t)^\top \frac{\partial f_\theta}{\partial \theta}(t, y(t)). \tag{5.1}
$$

Equations (5.1) are known as the continuous adjoint equations. 

$\frac{dL}{dy(T)}$ can be calculated from the output layer by auto differentiation.  Then the system is solved backwards in time to compute $\frac{dL}{d\theta}$. 

Need $y$ as an input, solve ode backwards in time using current $\theta$. 

$$
$$

$$
y(t) = y(T) + \int_t^T f_\theta(s, y(s))\, ds. \tag{5.2}
$$
Often referred to as the *adjoint method*, however this can also refer to other methods. 
More often referred to as the continuous adjoint method or optimise then discretise. 



## Existence and Uniqueness:

## Stability 

For NODE's spectral norm conditions keep trajectories bounded: described in Kidger ?


# Neural Controlled Differential Equations: 

>[!quote] Neural ODEs were the continuous-time limit of residual networks. We will now introduce neural controlled differential equations as the continuous-time limit of recurrent neural networks. - (Kidger, 2021).

[[Controlled Differential Equations (CDEs)]]

$$
z(t) = z(0) + \int_0^t f_{\theta}(s, z(s))\, dX(s) \quad \text{for } t \in (0, T]. \tag{1}
$$
is a Riemann-Stieltjes integral - allowing for non-smooth paths. 

# Neural Stochastic Differential Equations

$$
z(t) = z(0) + \int_0^t f(s, z(s); \theta_f )\, ds + \int_0^t \ g(s, z(s); \theta_g) \ dW(s) 
$$

where W(t) is a Wiener process. 

The integral can follow either Ito or Stratonovich formulations. 


#### Important Literature
First introduction of NODEs[[chenNeuralOrdinaryDifferential2018 |  Chen et al., (2018) ]]
Textbook like PhD Thesis: [[kidgerNeuralDifferentialEquations2022| | Kidger (2022) ]]


