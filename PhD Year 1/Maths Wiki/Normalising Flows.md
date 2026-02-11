
Start with a simple distribution and want to transform it into a more complex target distribution. A comprehensive review paper can be found in (Kobyzev, Prince & Brubaker, 2021).

Want to find invertible functions which take a sample distribution, $Z$ and bring to a target distribution $X$. So we want to find a function $f: Z \to X$.  
We do this through a "normalising flow" of repeated distributions. 
$$
x = f(z) = f_{K} \; \circ f_{K-1} \; \circ \; ... \; \circ \; f_2 \; \circ f_1 (z)
$$
We know that the distribution of 

	$$ p_{\theta}(x) = p_\theta(x)  \left| det \left( \frac{  \partial f^{-1}}{\partial x} \right) \right| = p_{\theta}(z) \prod^N_{i = 1} \left| det \left( \frac{  \partial f_i^{-1}}{\partial x} \right) \right| $$
	