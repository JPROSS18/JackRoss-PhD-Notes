***

Method of gradient computation in a neural network/ 

# Notation:

For a Feedforward Neural Network with $L$, loss function $\mathcal{L}$, input vector $x \in \mathbb{R^n}$ and a target vector $y \in \mathbb{R}^m$.

Let $W^\ell$ be a matrix of weights between the layer $\ell$ and $\ell - 1$ and $\sigma_{\ell}$ is a non-linear activation function. 
Let
$$
\begin{aligned}
a_1 = \sigma_1 (W^1 x) \\
a_\ell = \sigma_\ell(W^{\ell}a_{\ell-1})
\end{aligned}
$$

$\hat y = a_L$, 

$$
\hat y = \sigma_L ( W^L \ ....  \ \sigma_2 ( W^2 \sigma_1(W^1 x) \ .... \ )
$$
The goal is minimise: 
$$ 
\mathcal{L}(y , \hat y)
$$

Let $\theta$ be a vector of the weights $\{ w^{\ell}_{jk} \}$.

$$
\theta_{i+1} = \theta_i + \nabla_{\theta_i} \mathcal{L}(y, \hat y) 
$$
Backpropagation is an efficient method to compute $\nabla_{\theta_i} \mathcal{L}(y, \hat y)$.


# Description:

We know $a_{\ell} = \sigma^{\ell}(W^{\ell}a_{\ell - 1} + b^{\ell})$. 

Let $a_\ell = f^{\ell}(a_{\ell -1})$. 

Passing fixed $x$ through the network computes $\{a_\ell\}^{L}_{\ell =1}$ . This is the forward pass. 



To calculate the gradient: it is necessary to compute $\frac{\partial \mathcal{L}}{\partial w^{\ell}_{ij}} \in \mathbb{R}$ for all weights. 
 $\frac{\partial \mathcal{L}}{\partial W^{\ell}}$ is the vector of partial derivative w.r.t to the weights connecting layer $\ell$ and $\ell - 1$. 

From the chain rule we can see that: 
$$
\frac{\partial \mathcal{L}}{\partial W^{\ell}} \ = \frac{\partial \mathcal{L}}{\partial a_L} \frac{\partial a_L}{\partial a_{L-1}} .... \frac{\partial a_{\ell + 1}}{\partial a_\ell} \frac{\partial a_{\ell}}{\partial W^{\ell}}
$$
as $a_L$ depends on $W^{\ell}$ only through $a_\ell$. Direct computation for all weights would be inefficient $O(L^2)$. 

Note that there are many repeated calculations. Define: 

$$
 \alpha^{(\ell)} \ = \ \frac{\partial \mathcal{L}}{\partial a_\ell} = \frac{\partial \mathcal{L}}{\partial a_L} \frac{\partial a_L}{\partial a_{L-1}} .... \frac{\partial a_{\ell + 1}}{\partial a_\ell}
$$

 Note that $\alpha^{(\ell)}$ is vector with dim equal to the dimension of $a_\ell$ (product of matrices and $\mathcal{L}$ is a scalar function). 

$\alpha^{(\ell)}$ can but computed recursively backwards.
$$
\alpha^{(\ell - 1)} = \alpha^{\ell} \frac{\partial a_\ell }{\partial a_{\ell-1}}
$$
Then we can compute: 
$$
\frac{\partial \mathcal{L}}{\partial W^{\ell}} = (\alpha^{(\ell + 1)})^T \frac{\partial a_{\ell + 1}}{\partial W^{\ell}}
$$
The computation of the pairs $(\alpha^{(\ell + 1)}, \frac{\partial a_{\ell + 1}}{\partial W^{\ell}})$ is called the backward pass. 
# **History:**
Multiple discovers: 
*Reverse mode automatic differentiation/reverse accumulation*

Has it origins in 1950s Optimal control theory. 

>[!quote] The first [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron "Multilayer perceptron") (MLP) with more than one layer trained by [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent "Stochastic gradient descent")[[20]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-robbins1951-29) was published in 1967 by [Shun'ichi Amari](https://en.wikipedia.org/wiki/Shun%27ichi_Amari "Shun'ichi Amari").[[26]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-Amari1967-35)

The MLP had 5 layers, with 2 learnable layers, and it learned to classify patterns not linearly separable


In 1982, [Paul Werbos](https://en.wikipedia.org/wiki/Paul_Werbos "Paul Werbos") applied backpropagation to MLPs in the way that has become standard.[[31]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-werbos1982-40)[[32]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-werbos1974-41) Werbos described how he developed backpropagation in an interview. In 1971, during his PhD work, he developed backpropagation to mathematicize [Freud](https://en.wikipedia.org/wiki/Sigmund_Freud "Sigmund Freud")'s "flow of psychic energy". He faced repeated difficulty in publishing the work, only managing in 1981.[[33]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-:1-42) He also claimed that "the first practical application of back-propagation was for estimating a dynamic model to predict nationalism and social communications in 1974" by him.[[34]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-43) what?
