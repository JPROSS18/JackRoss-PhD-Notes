***

**History of Machine Learning Prediction:**
2020:  A scripted video of Obama required 2h of audio, 50 hours of video and 15k of compute time. 
2025: Done live in Class. Clones himself in real time. 

**Deep Learning:** Machine learning which employs artificial neural networks. 


Artificial Intelligence -> Machine Learning (No explicit programming) -> Deep Learning (Using deep neural networks). 



Why Now: 

Neural networks have existed for a long time, why now? 

1. Big Data: Availability of data has massively increased in recent years. 
2. Hardware: New Hardware such as GPU enables efficient Parallel computing. 
3. Software: Improved software leads to improved research.



|      | Timeline of ML                          |
| ---- | --------------------------------------- |
| 1952 | Stochastic Gradient decent              |
| 1958 | Perceptron: Learnable Weights           |
| 1986 | Backpropagation + Multilayer Perceptron |
| 1995 | Deep Convolution NN: Digit Recognition  |
|      |                                         |


##### The Perceptron: Forward Propagation

 ![[Mit_DL_L1_D1]]
 Written as single function: Weighted Sum through a non-linear activation function. 
$$ \hat y = g\left( w_0 + \sum^{m}_{i=1}w_i x_i\right) $$
Rewritten through linear algebra: Letting

$$X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}, \; W = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}$$
$$ \hat y = g\left( w_0 + X^T W \right)$$
##### Activation Functions: 
![[Pasted image 20251023175007.png]]
Why are they needed:
Real data is non-linear, NN would be linear otherwise. NN would be reduced to linear regression. 
With enough depth a NN can approximate any function


##### From Perceptions to Neural Networks

![[Mit_DL_L1_D2]]
Adding a second node. It is called **Dense** as all nodes are connected to all nodes in previous layer. 

##### Code: Dense Layer in Code

Pytorch: 

##### Hidden Layers: 

When layers are added between input and output they are called hidden layers. A deep network has many different layers. 
![[Pasted image 20251023181058.png]]

More depth -> More potential for non-linearity
More Width -> How many variable considered - dimensionality of problem. 

##### Applying Neural Networks: 

Problem given: as will I pass this class? 

We have yet to train a setwork - introducing loss functin 

##### Loss and Cost Function:

The **loss function**, $\mathcal{L}$, quantifies how close a predicted output, $\hat y$, is the the actual output $y$. 
$$ \mathcal{L}\left(f(x^{(i)}; W ), y^{(i)})\right)$$

The **cost function** takes the average of the sum of the loss function evaluated on all available training data. 

$$ C(W) = \frac{1}{n}\sum^n_{i=1} \mathcal{L}\left(f(x^{(i)}; W ), y^{(i)})\right)$$
Various loss functions exist. Correlation cross entropy is common for classification. MSE is common for prediction. 

###### Training Neural Networks:

We want to select optimal weights to minimise the cost function. 

Basic idea is **gradient decent**. 
1. Step 1: 
Starting with random weights. Take the gradient of cost function w.r.t weights. 
$$\frac{\partial C(\textbf W)}{\partial \textbf W}$$
2. Step 2: 
Take the negative of the gradient from the Weights. 
$$ W \leftarrow W - \eta \frac{\partial C(\textbf W)}{\partial \textbf W}$$
This points in the direction of a local minimum (gradient points in the the direction of steepest incline). 
3. Repeat until you reach a local minimum and return weights. 


##### Backpropagation
Gradient decent can be computationally heavy. An algorithm called [[Backpropagation]] is used to compute the gradient efficiently 

**How does backpropogation work?**

![[MIT_DL_L1_D3]]

##### Optimisation: 

During gradient decent, the parameter $\eta$ is the learning rate. It controls how much the weights change at each step. Important not to jump over local min/ get stuck. 

$$ W \leftarrow W - \eta \frac{\partial C(\textbf W)}{\partial \textbf W}$$
Many algorithms exist to implement adaptive learning rate. One common popular algorithm is [[Adaptive Moment Estimation (Adam)| Adam]]. 

##### Stochastic Gradient Decent: 

More commonly used. Gradient will be computed on one data point $x_i$, instead of full data set. Much faster but is noisy. 

Mini-batch gradient controls the level of noise in the gradient by computing gradient on $k$ randomly selected data points. Higher $k \to$ less noise but more computation. 

Training can be parallelised, each data point independent !

##### Overfitting
It is important that models generalise the unseen data well and does not "overfit" to training data - model must not learn the noise present in training data.
If we "overfit", the test accuracy will reduced even when model captures training data fully. 

If we force the model to think to much about a simple problem (more complexity/non-linearity than necessary), it will notice patterns that don't exist, similar to us.

##### Regularisation 

Method to prevent overfitting. 

**Dropout:** Randomly selection nodes on each layer to drop on each step during training. This adds some stochasticity, ensuring the network does not simply memorise the training data (the same network will not shown the same data repeatedly) 

**Early Stopping:** Monitor the deviation between training loss and the loss on a "holdout" set. 
