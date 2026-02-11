

| Field              | Value                                                                                                          |     |
| ------------------ | -------------------------------------------------------------------------------------------------------------- | --- |
| **Title**          | Neural Ordinary Differential Equations                                                                         |     |
| **Year**           | 2018                                                                                                           |     |
| **Authors**        | Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud                                                |     |
| **Tags**           | Computer Science - Artificial Intelligence, Computer Science - Machine Learning, Statistics - Machine Learning |     |
| **Citation Key**   | chenNeuralOrdinaryDifferential2018                                                                             |     |
| **Zotero PDF**     | [Preprint PDF](zotero://select/library/items/SKK7HIGC)                                                         |     |
| **Related Papers** | [[kidgerNeuralDifferentialEquations2022]]                                                                      |     |

---

---

Related::  [[kidgerNeuralDifferentialEquations2022]] 

# Persistent Notes

## Outline
Introduced the concept of [[Neural Differential Equations (NDE's) | Neural ODE's]] as the continuous limit of a [[Residual Neural Network | residual network]] or a [[Normalising Flows | normalising flow]]. 

They reference a series of papers to justify that resnet/normalising flows can be seen as a Euler discretisation of an ODE (Lu et al., 2017; Haber and Ruthotto, 2017; Ruthotto and Haber, 2018). 

$$ h_{t+1} = h_t + f(h_t, \theta_t) $$
Each hidden layer is derived by added a specified transformation to the previous layer. 

Taking the continuous limit limit: 

$$ \frac{d\textbf{h(t)}}{dt} = f(\textbf{h}(t), t, \theta) $$
where the input layer is $h(0)$ and the output layer is $h(T)$. 


## Benefits of Approach: 

The authors outline the benefits of this approach within the introduction. 

**Improved memory efficiency:** Using the *adjoint sensitivity method* (optimise then discretise). Intermediate quantities are not stored, only need current state. Memory needed is a constant function of the depth  (previous states are needed in non-feedforward neural networks??).

![[NODE Sketch 21-10]]

**Adaptive Computation:** Rich theory of numerical analysis exists, so guarantees of error can be made when solving using well-studied solvers (RK4 etc.). Can have an explicit computation cost/accuracy trade off when choosing ODE solver. 

**Parameter Efficiency:** As the hidden layers are parameterised as a continuous function nearby points are automatically impacted by each other, leading to efficiency when training **(Is this the case in both for all training schemes ? Or just optimise then discretise)**. 

**Scalable and invertible flows: ** The authors claim that due to continuity, the flow is more easily invertible. Therefore avoid some bottlenecks seen in discrete [[Normalising Flows | normalizing flows.]].

Neural ODE's also function as a more natural model of cts dynamics.


#### Training the ODE
Reverse-mode automatic differentiation of ODE

How to conduct [[Backpropagation]] through an ode


%% end notes %%

### In-text annotations



%% Import Date: 2025-10-21T12:54:31.824+01:00 %%
