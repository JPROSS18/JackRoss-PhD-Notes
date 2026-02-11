The correlation dimension is a measure of dimensionality of a set. Like box-counting dimension and Hausdorff dimension, it is a type of fractal dimension. 

For a trajectory n n-dimensional space: 


$$
\{x_1, x_2, ..., x_N\}, x_i \in \mathbb{R}^n
$$


# Hausdorff Dimension
Hausdorff dimension - most natural measure of dimension to me  - like a measure space. Comes from the growth of number of cubes needed to cover the grows exponentially. Geometric.

The compute the Hausdorff dimension, the attractor is covered by n-dimensional hypercubes of side length $\ell$  and the limit $\ell \to 0$ is considered. 
Let $M(\ell)$ be the minimum number of cubes needed for a covering of the attractor. As $\ell \to 0$, 
$$
M(\ell) \simeq \ell ^{-d}
$$

# Information Dimension

The information is a dimension is closely related to [[entropy | entropy]] . The information dimension measures the growth rate of the Shannon entropy given successively smaller discretisation of the space. 


$$
S(\ell) = - \sum^{M(\ell)}_{i = 1}p_i\ln(p_i)
$$
where $p_i$ is the probability that a randomly selected point $x_j$ falls in the $i$th interval.

It a lower bound of the Hausdorff dimension. 

# Correlation Dimension

A measure of the dimension of an attractor obtained from the correlations between points. First introduced in (Grassberger & Procaccia, 1983). 

The compute $d_{corr}$, first it is necessary to compute the correlation integral, which can be computed as: 
$$
C(\varepsilon) = \lim_{N \to \infty} \frac{1}{N^2} \sum^N_{i \neq j, \ j = 1} \sum^N_{i = 1} \Theta(\epsilon - ||x_i - x_j ||)
$$


where $\Theta$ is the Heaviside step function. 

The has been shown that for small $\varepsilon$ , $C(\varepsilon) \sim \varepsilon^\nu$. Where $\nu$ s the correlation dimensions, $d_{corr}$. 

$$
d_{corr} \leq d_{inf} \leq d_{haus}
$$


The correlation dimension is the computed as: 

$$d_{corr} = \lim_{\varepsilon \to 0}\frac{\ln(C(\varepsilon))}{ln(\varepsilon)}

$$


## Computing the Correlation Dimension


