Show how old dynamical techniques can be made data driven - which are usually left in data analysis 
Nathan Kutz/Steve Brunton

Lot of theory about when dynamics can be recast into simpler settings (prove a function exists). Neural Networks can now find these functions. 

> [!quote] The goal has always been to showcase a suite of computational methods that can be combined with analysis to better understand the increasingly complicated models and datasets that describe our complex world

This means that I have neglected certain methods, such as diffusion maps, since from my perspective it has fallen out of favor by the community and has been replaced with more robust machine learning approaches that are covered in the book.


>[!quote] Similarly, the emerging methods surrounding neural differential equations have been omitted since they are primarily for forecasting dynamical systems. While interesting in their own right, forecasting dynamics is slightly away from the focus of this book as it does not rely on exploiting any dynamical features of the system, making it more closely related to numerical analysis than dynamical systems theory. That is, this book is mostly concerned with interpreting data from a dynamical systems perspective, meaning classifying features such as fixed points, periodic orbits, chaotic attractors, invariant measures, and so on.

**. In this book we will bring together computational tools such as neural networks, sparse regression, dynamic mode decomposition, and semidefinite programming to provide an accurate understanding of dynamic data.** - Main goal 

Study systems which change in time. 

Discrete: 
$$
x_{t+1} = f(x_t) 
$$
Continuous: 
$$
\dot x = f(x, t)
$$


What is Dynamical System maths 
> [!quote] Precisely, dynamical systems analysis emphasizes a geometric interpretation of the system. Classical analysis of differential equations at least starts with existence and uniqueness theorems, while the geometric intuition of dynamical systems might emphasize the existence of invariant manifolds in phase space; see Figure 1.1 for a comparison.

Standard analysis will consider existence and uniqueness / analytical/ numerical solutions. 

Dynamical systems will take a more geometric approach, consider invariant manifolds. (dependent on/independent of initial conditions)

Knows that a solutions exists -> tells us nothing about the solution. 

>[!quote] Many questions in dynamical systems revolve around how solutions evolve in phase space, particularly where they are going and how they get there.

. This is where the geometric perspective comes in. One is interested in manifolds embedded in phase space that are invariant with respect to the flow of a system and what potential limit sets look like.

Talks about lorenz systems eigenvalues. 

The stable manifold theorem guarantees that when the origin is stable (all negative eigenvalues). all initial conditions that are chosen sufficiently close to the origin will result in trajectories satisfying (x1(t), x2(t), x3(t)) → (0, 0, 0) as t → ∞ [109].

Dimension of the stable invariant manifold corresponds to the number of negative eigenvalues. 

**What is included in this book** 

Beyond the local stability results outlined here, this further includes identifying **transformations to canonical forms, finding invariant measures that describe the probability of finding trajectories in certain regions of space, and classifying the types of invariant manifolds that may be present in a system**. 

**Area of Maths used:** 
From just this list alone we find that dynamical systems incorporates ideas from not only geometry but also **algebra (canonical forms),** **probability** and **ergodic theory (invariant measures)**, and **topology (invariant manifolds),** to only name a few. Therefore, dynamical systems analysis presents itself an application of what are sometimes viewed as the most abstract areas of mathematics.

Through dynamical systems there are applications of the purest areas of mathematics. 

# 1.2 Influence of Computers

Computer means that new physical effects can be modelled that would previously have been impossible to analyse. 

Allowed previously theoretical ideas possible -> chaos (poincaré)- erdogic theory - Random variables. 

Lorenz System - Deterministic Non-periodic flow (1963)

The ability to **“see”** the trajectories of differential equations in phase space is no better typified than in the Lorenz system (1.1). With the help of E. Fetter and M. Hamilton, 

Lorenz simulated trajectories of his system to visualize their movement through the three-dimensional phase space. The results were staggering, as you can see for yourself in Figure 1.2. Lorenz and his team found that solutions evolved in a very complex manner, tracing out what is now known as the butterfly attractor.

Take a while to formulate what is actually meant by chaotic - Poincaré 

>[!quote] ‘Special thanks are due to Miss Ellen Fetter for handling the many numerical computations’

Margret Hamilton - develop some of the first weather predicting software. 

She left in 1961 - Replaced by Ellen Fetter 
- First person to use term 'Software Engineering'. 
- Apollo flight lead software designer

Really showed the potential of scientific computational modelling.


Can provided a first step to theoretical understanding. 

# Proper Orthogonal Decomposition

[[Proper Orthogonal Decomposition (POD)]]

# Linear Evolution models

Models of evolving dynamical systems by linear 

[[Dynamic Mode Decomposition (DMD)]]
