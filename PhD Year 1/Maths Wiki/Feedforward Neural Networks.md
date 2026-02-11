

# Thoughts

These are hard to train due to the multiplicative amplification of signals propagating through the layers, causing signals to either explode or vanish in magnitude if the number of layers is too large.

Mathematical analysis in the asymptotic limit of infinitely wide layers reveals how deep networks can nevertheless learn to solve classification tasks [4–7]. Recent results indicate that finite-width networks may learn in different ways [8–10]. 


Training as a dynamical system: 
$$
\theta_{i+1} = \theta_i + \nabla_{\theta_i}L(\bf{x})
$$
Autonomous dynamical system
>[!quote] It is not understood, however, when and how such networks use their exponential expressivity to represent data features needed for a classification task, how the representation affects prediction accuracy and uncertainty, and how it depends on the network layout. (Storm et al., 2024) 

# Gradient Descent as a Dynamical System

Training as a dynamical system: 
$$
\theta_{i+1} = \theta_i + \nabla_{\theta_i}L(\bf{x})
$$
Autonomous dynamical system for standard loss functions (L1/L2)

# Lyapunov Exponents of Deep NN

If network weights $w^{(\ell)}_{ij}$ are initialised by gaussian distribution with zero mean and variance $\sigma^{2}_{\ell}$. 

Lyapunov exponents are determined by a product of random matrices. 

At layer $\ell$ , the output is: 
$$
x^{(\ell)}_{i} = g(\sum^{N_{\ell}}_{j = 1}w_{ij}^{(\ell)}x^{\ell - 1}_j - b^\ell_i ) 
$$
A discrete dynamical system with $\ell$ function as time.  (Non autonomous)

As $L \to \infty$, $\lambda^{(L)}$ converges and for $N \to \infty$, $\lambda^{(L)} ~ log(GN\sigma^2)$.  

(Storm et al., 2024) conjectures that 


> [!quote] The ridges gradually disappear as the number N of hidden neurons per layer increases, because the maximal singular value of JLðxÞ approaches a definite x-independent limit as N → ∞ at fixed L. But how can the network distinguish inputs with different targets in this case, without ridges indicating decision boundaries? One possibility is that the large number of hidden neurons allows the network to embed the inputs into a high-dimensional space where they can be separated thanks to the universal approximation theorem [40]. In this case, training only the output weights (and threshold) suffices, as demonstrated by Fig. 3(a). That the classification error with random hidden weights is larger than that of the fully trained network is not surprising, since different random embeddings have different classification errors when the number of patterns exceeds twice the embedding dimension [11,41]. In summary, large-N networks can classify with random hidden weights. This implies that the hidden weights do not need to change  during training, just as in kernel regression with the neuraltangent kernel [4,18]. In other words, the learning in this regime is lazy [19]. - (Storm et al., 2024) 

A low dimensional network learns to classify the data in a fundamentally different way than a high dimensional one. Low dimensional network updates weights significantly to mimic dynamics of the data, whereas high dimensional systems embed the data and focus on output training (lazy). Seen in (Storm et al, 2024), where fixed weights (trained output layer) and trained weights perform similarly. 

