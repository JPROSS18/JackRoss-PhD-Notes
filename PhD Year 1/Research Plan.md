# Scientific Machine Learning for Non-Autonomous Dynamical Systems. 

In recent years, scientific machine learning has revolutionised the potential of data-driven modelling in scientific study. 

For decades, dynamical systems theory has provided the theoretical foundation for for the mathematical study of time varying system.s 

However, the interaction between these fields has often been limited. Im particular, using simplier machine learning architectures, such as reservoir computers. 

## Non-autonomous Dynamical Systems
We hope to 
1. Apply state of the art machine learning architectures in order to reconstruct dynamical systems directly from data. 
2. Use dynamical systems theory to introduce methods to quantitively assess the performance of ML methods in reconstructing dynamical systems beyond simple L2 loss. 

In particular, we hope to focus on *non-autonomous dynamical systems*.

1. Optimise current machine learning architectures for learning of non-autonomous systems. 
2. Use existing dynamical systems theory in order to study the dynamics of reconstructed systems. 

## Critical Transitions 
A particular focus should be paid to the study of critical transitions, a where a qualitive change in system dynamics occurs. 

The study of critical transitions is central of to the mathematical study of many real world systems. 

The study of critical transitions using purely data-driven approaches is diffucult as the systems changes between states where the dynamics can be highly different. 

Additionally, exisitng methods for the analysis of critical transitions -(early warning signals and related), are difficult int he non-autonomous case due to limited data. 

We hope to investigate if machine learning approaches can reconstruct non-autonomous dynamical systems experiencing *dynamic bifurcations*, with possible extensions to *noise* and *rate-induced* critical transitions.  

# Neural ODE's

Neural Differential Equations are a natural extension of modern machine learning methods to classical mathematical modelling methods. 

The study of neural ODE's is not often used in the non-autonomous case. As it involves learning a time-dependent vector field, rather than a static one. 

The study of critical transitions from a machine learning perspective has historically been limited ot reservoir computers. However, this is limited to bifurcation parameters which are non-time dependent, a limiting the problem to learning multiple autonmous systems rather than a single non-autonomous one. 

Additionally, as a reservoir computer is inherently autonomous, its is non-trivial how to extend this architecture of non-autonomous problems.

However, neural ODE's are a more natural choice for learning dynamic critical transitions. However, this area is too understudied. 

1. Introduce optimal neural differential equations architectures for learning non-autonomous systems, in particular systems which undergo a critical transition. 
2. Intro quantitive methods from based on dynamical systems theory which assess the reconstructed dynamics in the neural differential equation 

## Sleep-onset Period
Additionally, we intend to explore the direct application of this work to the study of the sleep on-set period. This noisy non-autonomous system which models the transition from wake to sleep in humans. This data will be used to assess the effectiveness of the our methods. 

