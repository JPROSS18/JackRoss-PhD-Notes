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

**Learning Slow Dynamics:**
$$
\begin{aligned}
& \dot r = \alpha r + (1 - \alpha)\tanh(Mr + W_{in} x  + b) \\
\end{aligned}
$$
Take a reservoir with a very high leak rate ($\alpha \approx 0.995$). 

Take a moving average each node $r_i$, then calculate the standard deviation around each node. Node with smallest standard deviation correspond the slowly varying dynamics. 

Call these nodes $r^{s}_i$ and take the average of the absolute values at each $t$

$$
r^s = \frac{1}{N_s} \sum_{i \in S} |r^s_i|
$$
Then as a linear filter to $r^s$ and feed into fast reservoir. 

**LInear Filter:**




