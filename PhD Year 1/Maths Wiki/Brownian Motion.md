*Brownian Motion* was first observed by Robert Brown as the irregular movement of pollen grains suspended in water in 1828. It was later formalised as a stochastic process by Einstein. 

# Definition

Let $(\Omega, \mathcal F, P)$ be a probability space with a filtration $\{ \mathcal F_t \}_{t \geq 0}$. A one-dimensional Brownian motion is a real-valued continuous $\{ \mathcal F_t \}$-adapted process $\{ B_t \}_{t \geq 0}$ with the following properties:

1. $B_0 = 0$ a.s. 
2. $0 \leq s < t < \infty$, $B_t - B_s$ is normally distributed with mean zero and variance $t-s$.
3.  $0 \leq s < t < \infty$, $B_t - B_s$ is independent of $\mathcal F_s$.


The increments $B_t - B_s$ are stationary and independent. 

$\mathcal F^B_t = \sigma( B_s : 0 \leq s \leq t)$ for $t \geq 0$ is called the *natural filtration*. 

# Properties

1. $\{ - B_t \}$ is a brownian motion. 
2. For $c \geq 0$. 
$$
X_t = \frac{B_{ct}}{\sqrt{c}} \quad t \geq 0
$$
is a Browian motions w.r.t $\mathcal F_{ct}$, 
3. $\{ B_t \}$ is a continuous square integrable martingale and its quadratic variation, $\langle B, B\rangle_t = t$ for all $t \geq 0$.
4. The strong law of large numbers states that:
$$
\lim_{t \to \infty} \frac{B_t}{t} = 0, \quad a.s.
$$
5. For almost every $\omega \in \Omega$, the Brownian sample path is nowhere differentiable. 
6. For almost every $\omega \in \Omega$, the sample path is locally Holder continuous with exponent $\delta \in (0, 1/2)$. However, it is nowhere Holder continuous for $\delta > 1/2$