

| Field              | Value                                                                                                                          |     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------ | --- |
| **Title**          | Augmented Neural ODEs                                                                                                          |     |
| **Year**           | 2019                                                                                                                           |     |
| **Authors**        | Emilien Dupont, Arnaud Doucet, Yee Whye Teh                                                                                    |     |
| **Tags**           | Computer Science - Machine Learning, Statistics - Machine Learning                                                             |     |
| **Citation Key**   | dupontAugmentedNeuralODEs2019                                                                                                  |     |
| **Zotero PDF**     | [Full Text PDF](zotero://select/library/items/FVPE2TT4)                                                                        |     |
| **Related Papers** | [[kidgerNeuralDifferentialEquations2022]], [[tegelenNeuralOrdinaryDifferential2025]], [[chakrabortyDivideConquerLearning2024]] |     |
|                    | [[Neural Differential Equations (NDE's)]]                                                                                      |     |

---

---
Zotero PDF Link: 
Related:: 

### Notes

Weight decay vs ANODE?

ANODE outperforms WD in terms of loss + computational cost. However, total computational cost decreased when both are combined, at the cost of increase in loss. 

Stability: NODE can sometime create very illposed odes, which lead to tiny prescision needed by odesover. 

Scaling: ANODE achieve lower loss + train faster. 

Problems addressed by ANODE can also solved by using CNN instead of MLP in NODE. However, this increasing the complexity of the model in way that ANODE does not. 

Similar approaches used for Resnets improve performance. However, they do not reduce computation cost (as resnets can been as euler scheme, so number of computations remains constant.)

Examples can be found at https://github.com/EmilienDupont/augmented-neural-odes. 
### In-text annotations

 <mark class="hltr-red">"Figure 3: (Top left) Continuous trajectories mapping −1 to 1 (red) and 1 to −1 (blue) must intersect each other, which is not possible for an ODE. (Top right) Solutions of the ODE are shown in solid lines and solutions using the Euler method (which corresponds to ResNets) are shown in dashed lines. As can be seen, the discretization error allows the trajectories to cross. (Bottom) Resulting vector fields and trajectories from training on the identity function (left) and g1d(x) (right)."</mark> [Page 3](zotero://open-pdf/library/items/FVPE2TT4?page=3&annotation=X9UPNJKL)

Resnet learns as error allow trajectories to cross- but this is a limit of odes in general, not just neural odes. [Page 3](zotero://open-pdf/library/items/FVPE2TT4?page=3&annotation=X9UPNJKL)


 <mark class="hltr-red">"Proposition 3. The feature mapping φ(x) is a homeomorphism, so the features of Neural ODEs preserve the topology of the input space.  A proof is given in the appendix. This statement is a consequence of the flow of an ODE being a homeomorphism, i.e. a continuous bijection whose inverse is also continuous; see, e.g., (Younes, 2010). This implies that NODEs can only continuously deform the input space and cannot for example tear a connected region apart."</mark> [Page 3](zotero://open-pdf/library/items/FVPE2TT4?page=3&annotation=XW96NYPP)

Very interesting [Page 3](zotero://open-pdf/library/items/FVPE2TT4?page=3&annotation=XW96NYPP)




%% Import Date: 2026-02-01T16:20:19.703+00:00 %%
