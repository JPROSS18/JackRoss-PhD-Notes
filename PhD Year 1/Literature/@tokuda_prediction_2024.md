---
title: Prediction of unobserved bifurcation by unsupervised extraction of slowly time-varying system parameter dynamics from time series using reservoir computing
authors: Keita Tokuda, Yuichi Katori
year: 2024
DOI: 
Zotero: zotero://select/items/@tokuda_prediction_2024
---

Notes that [[@patelMachineLearningPredicting2024]] also deals with dynamic bifurcations. Predicts unobserved bifurcations using reservoir computer when $r(t)$ is known. 

The authors of this paper attempt to extend this approach by to the case of unknown $r(t)$ by using two seperate reservoirs with different time scales. 

For case where timescale of $\dot r$ is much smaller than timescale of actual system. 

$$
\begin{aligned}
& \dot r = \alpha r + (1 - \alpha)\tanh(Mr + W_{in} x + W_{param}\lambda + b) \\
\end{aligned}
$$
$\lambda$ is learned by another reservoir 

Very good papers - include in literature review 