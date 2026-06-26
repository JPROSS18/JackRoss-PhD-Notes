(This page was created with the aid of generative AI)
# Autoregressive Models

Autoregressive models are a class of time series models where the current value of a series is expressed as a linear function of its own past values plus some noise term.

## The General AR(p) Model

An autoregressive model of order $p$, written AR(p), takes the form:

$$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \varepsilon_t$$

where:

- $c$ is a constant
- $\phi_1, \ldots, \phi_p$ are the model coefficients
- $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ is white noise

The idea is simply that today's value is a weighted sum of the previous $p$ values, plus a random shock.

## The AR(1) Model

The simplest case is AR(1):

$$X_t = c + \phi X_{t-1} + \varepsilon_t$$

The single parameter $\phi$ controls everything interesting about the model's behaviour:

|Condition|Behaviour|
|---|---|
|$\|\phi\| < 1$|**Stationary** — shocks decay, series reverts to mean|
|$\phi = 1$|**Random walk** — shocks are permanent|
|$\|\phi\| > 1$|**Explosive** — shocks grow over time|

## Mean and Variance

For a stationary AR(1) where $|\phi| < 1$, taking expectations of both sides gives the mean:

$$\mu = \mathbb{E}[X_t] = \frac{c}{1 - \phi}$$

The variance is:

$$\text{Var}(X_t) = \frac{\sigma^2}{1 - \phi^2}$$

> [!note] The variance blows up as $\phi \to 1$, consistent with the random walk case being non-stationary.

## Autocorrelation

The autocorrelation at lag $k$ — the correlation between $X_t$ and $X_{t-k}$ — decays geometrically:

$$\rho_k = \phi^k$$

For example, with $\phi = 0.8$:

$$\rho_1 = 0.8, \quad \rho_2 = 0.64, \quad \rho_3 = 0.512, \quad \ldots$$

This geometric decay in the autocorrelation function is the signature that identifies an AR(1) process in real data.

## Estimation

$\phi$ and $c$ are estimated by OLS regression of $X_t$ on $X_{t-1}$:

$$\hat{\phi} = \frac{\sum_{t=2}^{T}(X_t - \bar{X})(X_{t-1} - \bar{X})}{\sum_{t=2}^{T}(X_{t-1} - \bar{X})^2}$$

This estimator is consistent and asymptotically normal under stationarity.

## Intuition

A helpful way to think about AR(1) is as a noisy discrete dynamical system. Setting $c = 0$:

$$X_t = \phi X_{t-1} + \varepsilon_t$$

Each period, the series is "pulled" a fraction $\phi$ of the way toward zero, but then a random shock $\varepsilon_t$ kicks it off course. The tension between the mean-reverting pull and the random shocks generates the characteristic behaviour of stationary AR processes.