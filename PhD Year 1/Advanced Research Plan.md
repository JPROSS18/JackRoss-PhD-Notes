# Scientific Machine Learning for Non-Autonomous Dynamical Systems. 

In recent years, the advent of easily trainable neural networks has lead to a revolution in scientific machine learning has revolutionised the potential of data-driven modelling in scientific study. 

For decades, dynamical systems theory has provided the theoretical foundation for for the mathematical study of time varying system.

However, the interaction between these fields has often been limited. Im particular, using simplier machine learning architectures, such as reservoir computers. 

## Non-autonomous Dynamical Systems
We hope to 
1. Apply state of the art machine learning architectures in order to reconstruct dynamical systems directly from data. 
2. Use dynamical systems theory to introduce methods to quantitively assess the performance of ML methods in reconstructing dynamical systems beyond simple L2 loss. 

In particular, we hope to focus on *non-autonomous dynamical systems*.

1. Optimise current machine learning architectures for learning of non-autonomous systems. 
2. Use existing dynamical systems theory in order to study the dynamics of reconstructed systems. 

Can statistics, such as instantaneous Lyapunov exponents be used to measure the convergence of 
[@janosi_overview_2024] Finite Time Theory of Non-autonomous Dynamical Systems with a parameter drift. 

Can statistics, such as instantaneous Lyapunov exponents be used to measure the convergence of dynamics ?

[@louwLearningClimateDynamical2025] Rigorous approach to learning dynamical systems (from measure-theoretic perspective.)

Can we extend existing methods for autonomous system to a non-autonmous case.  [@bramburgerDeepLearningConjugate2021] Exploit normal forms and topological conjugacies to learn dynamical systems. 




## Critical Transitions 
A particular focus should be paid to the study of critical transitions, a where a qualitive change in system dynamics occurs. 

The study of critical transitions is central of to the mathematical study of many real world systems. 

The study of critical transitions using purely data-driven approaches is difficult as the systems changes between states where the dynamics can be highly different. 

In non-autonomous sytems critical transitions can be caused by three distinct mechanisms: 
- Noise-induced Tipping (N-Tipping)
- Rate-induced Tipping (R-Tipping)
- Bifurcation-induced Tipping (B-Tipping)

**EWS for Dynamic Bifurcations:**
There exists extensive literature surrounding analysis and prediction of critical transitions focus on B-Tipping (include discussion of relevant literature). Universally, these methods rely upon the loss of linear stability experienced by a stable equilibrium as it approach a bifurcation point. However, in the case of truly non-autonomous dynamic bifurcation, the *tipping* will be delayed by the *ghost* of the bifurcation point. This phenomenon is well studied (include citations - dynamic bifurcation book + Kuehn Book).  A through discussion of the case of systems with a slowly varying bifurcation parameter in detailed in [@kuehn_multiple_2015]. 

We hope to investigate if machine learning approaches can reconstruct non-autonomous dynamical systems experiencing *dynamic bifurcations*,

Significant Questions Exist:
1.  Can we recover the drift of a bifurcation parameter using existing multi-scale methods? 
2.  Can we exploit topological conjugacies/normal forms to bias a machine learning model to accurately predict a bifurcation point. Could such an approach detect a loss of a stable equilibrium even when the observed tipping is delayed. 

The existing theory generally using machine learning to directly extrapolate past bifurcation point. We hope to use dynamical systems theory to develop more reliable methods with uncertainity quanitifcation to complete this task. 



**Related Literature:** 

**Classic Dyanmical Systems:** 
[@wieczorek_rate-induced_2023] Rate Induced tipping 

[@thompson_predicting_2011] Climate Tipping as a noisy bifurcation, a discussion of dynamic bifurcations vs noise tipping. 

Early-Warning signals and Wishful thinking paper 



**Machine Learning + Tipping:** 
[@huangDeepLearningPredicting2024] Classifier trained on trajectories which experience rate induced tipping to predict if trajectory will experience tipping or not. 

[@tegelenNeuralOrdinaryDifferential2025] Uses to neural ode to learn ode with known bifurcation parameter for prediction of bifurcation parameter. 

[@patelMachineLearningPredicting2024] Uses reservoir computer driven by bifurcation parameter for prediction of critical transition. 

[@panahiUnsupervisedLearningAnticipating2025] Uses Reservoir Computer + Autoencoder for fully data driven prediction. 

[@tokuda_prediction_2024] Uses 2 Reservoir Computers for fully data driven predicting (slow vs fast).



# Neural ODE's

Neural Differential Equations are a natural extension of modern machine learning methods to classical mathematical modelling methods. 

The study of neural ODE's is not often used in the non-autonomous case. As it involves learning a time-dependent vector field, rather than a static one. 

The study of critical transitions from a machine learning perspective has historically been limited ot reservoir computers. However, this is limited to bifurcation parameters which are non-time dependent, a limiting the problem to learning multiple autonmous systems rather than a single non-autonomous one. 

Additionally, as a reservoir computer is inherently autonomous, its is non-trivial how to extend this architecture of non-autonomous problems.

However, neural ODE's are a more natural choice for learning dynamic critical transitions. However, this area is too understudied. 

1. Introduce optimal neural differential equations architectures for learning non-autonomous systems, in particular systems which undergo a critical transition. 
2. Intro quantitive methods from based on dynamical systems theory which assess the reconstructed dynamics in the neural differential equation 

**Idea:**
A selection of ideas
- Can we bias the network to preserve invariant measure to unsure topological conjugacy (or stronger). Compute Invariant Measure from Koopman Operator?

Related Literature:
[@chenNeuralOrdinaryDifferential2018] Introduced Neural ODE

[@liuNeuralSDEStabilizing2019] Introduce Stochastic Neural ODE

[@chakrabortyDivideConquerLearning2024] Neural ODE for chaotic dynamical systems. 

[[@wohrer_tracking_2026]] Tracking Finite Time Lyapunov Exponents to robustify Neural ODE's 

## Sleep-onset Period
Additionally, we intend to explore the direct application of this work to the study of the sleep on-set period. This noisy non-autonomous system which models the transition from wake to sleep in humans. This data will be used to assess the effectiveness of the our methods. 

Related Literature:

[@yangWakesleepTransitionNoisy2016]  Original model which model the SOP as a noisy bifurcation. 

[@liFallingAsleepFollows2025a] Deterministic Modelling of SOP cycle, use early warning signals to predict bifurcation point. 

[@huLearningBistableCortical] Paper which fits noisy bifurcation model to individual sleep trajectories. 



