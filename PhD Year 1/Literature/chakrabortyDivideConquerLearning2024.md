

| Field              | Value                                                                                                                |     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- | --- |
| **Title**          | Divide And Conquer: Learning Chaotic Dynamical Systems With Multistep Penalty Neural Ordinary Differential Equations |     |
| **Year**           | 2024                                                                                                                 |     |
| **Authors**        | Dibyajyoti Chakraborty, Seung Whan Chung, Troy Arcomano, Romit Maulik                                                |     |
| **Tags**           | Computer Science - Artificial Intelligence, Computer Science - Machine Learning                                      |     |
| **Citation Key**   | chakrabortyDivideConquerLearning2024                                                                                 |     |
| **Zotero PDF**     | [Full Text PDF](zotero://select/library/items/5LSTM7WP)                                                              |     |
| **Related Papers** | [[kidgerNeuralDifferentialEquations2022]] [[chenNeuralOrdinaryDifferential2018]]                                     |     |
|                    |                                                                                                                      |     |
|                    |                                                                                                                      |     |

---

---
Zotero PDF Link: 
Related:: 

### Notes
%% begin notes %%
Neural ODE's have trouble learning chaotic dynamical systems. 

The authors introduce a new training method to account for issues of non-convexity + exploding gradients when learning chaotic systems. 

The method breaks the trajectory into multiple, non-overlapping intervals while adding a loss penalty for discontinuity between intervals. The width of the intervals is selected of based on the Lyapunov time scale. This is refered to as the [[Multi-step penalty (MP) method ]]. 

The method performs well on short term dynamics but also for learning the invariant statistics that are hallmarks of chaotic dynamics. 

> [!quote] We remark that “learning” of a chaotic system is defined not simply by accurately predicting instantaneous trajectory over time, but more importantly by matching its invariant statistics at the attractors.

**Why is learning chaos in gradient descent hard?:**
Extreme sensitivity to perturbations translates to exploding gradients during training. Very non-convex loss landscape (so poor local min is grad descent is to constricted). 


>[!quote] For applications of NODE to chaotic dynamics, such options [traditional regularisation to avoid exploding gradients] are not viable, since the emergence of chaos in trained model predictions is a desirable property.





References gradient descent of ergodic properties. 


%% end notes %%

### In-text annotations



%% Import Date: 2026-02-02T11:14:14.397+00:00 %%
