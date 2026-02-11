

| Field              | Value                                                    |     |
| ------------------ | -------------------------------------------------------- | --- |
| **Title**          | Deep learning for predicting rate-induced tipping        |     |
| **Year**           | 2024                                                     |     |
| **Authors**        | Yu Huang, Sebastian Bathiany, Peter Ashwin, Niklas Boers |     |
| **Tags**           | Machine Learning, Rate-induced tipping                   |     |
| **Citation Key**   | huangDeepLearningPredicting2024                          |     |
| **Zotero PDF**     | [Full Text PDF](zotero://select/library/items/6PLHQUZL)  |     |
| **Related Papers** |                                                          |     |

---

---
Zotero PDF Link: 
Related:: 

### Persistent Notes
%% begin notes %%
==**Abstract**==
Critical slowing can help detect significant transitions between distinct states., when it is caused by a [[Bifurcation]] and if forcing is slow when compared with the systems internal timescale. 

However, in many real world situations this is not the case. A transition may occur because the forcing exceeds a critical rate - [[rate induced tipping]]. 

Some transitions may also be caused by random perturbations crossing an unstable boundary - [[noise induced tipping]]. 

Critical slowing down methods fail to distinguish the latter two methods from no tipping. #

This paper attempts to introduce a deep learning  method to give early warnings for rate induced tipping. They do so in three examples (), indicating rate-induced tipping and noise-induced tipping are predictable. 

They intend to show CSD cannot detect Ripping, then show a DL framework may be able to learn. 
**Introduction**

Explains the concept of [[CSD]] and why is an early warning indicator for a Bifurcation induced tipping point. Then explains that as the basin of attraction may remain invariant during [[R-tipping]], this indictor is not useful for detecting R-tipping. 

Notes that when forcing is below critical rate is below critical rate, noise can still cause a trajectory to tipp into another system. When noise and forcing join together, trajectories that differ only by noise before tipping may have different outcomes (predicting is very difficult)
They intialised examples of R-tipping 
==**Discussion**==
Notable lack of literature on predicting rate induced tipping. Established [[CSD]] framework cannot be used when characteristic timescale of the system is much slower than rate of forcing. 

The DL algorithm proposed can extract higher order information about how far away the system from equilibrium, this information can be used to give quantitative probabilities of R-tipping happing. 

> "It is worth noting that two limitations need to be addressed before applying our current DL model to provide ‘generic’ EWS for tipping points in natural systems. On one hand, we conducted predictions for R-tipping and N-tipping on three individual systems. On the other hand, besides R-tipping, there are other tipping phenomena such as global bifurcations2,36 and tipping transitions from non-equilibrium attractors37 that do not rely on changes to the local stability of equilibrium and would, hence, also not be captured by CSD"

Suggest that a universal model could be possible and suggest conditions necessary that I did not understand. 

==**Machine Learning**==
[[CNN]]-based in pytorch was used.

Used [[Layer-wise Relevance Propagation (LRP)]] for explainability  

%% end notes %%

### In-text annotations

 <mark class="hltr-yellow">"However, in many real-world situations, these assumptions are not met and transitions can be triggered because the forcing exceeds a critical rate."</mark> [Page 1556](zotero://open-pdf/library/items/6PLHQUZL?page=1556&annotation=NJYSUMAT)


 <mark class="hltr-yellow">"Moreover, random perturbations may cause some trajectories to cross an unstable boundary whereas others do not—even under the same forcing"</mark> [Page 1556](zotero://open-pdf/library/items/6PLHQUZL?page=1556&annotation=AEQ9P577)


 <mark class="hltr-yellow">"Our findings demonstrate the predictability of rate-induced and noise-induced tipping, advancing our ability to determine safe operating spaces for a broader class of dynamical systems than possible so far."</mark> [Page 1556](zotero://open-pdf/library/items/6PLHQUZL?page=1556&annotation=AARVR34U)


 <mark class="hltr-yellow">"Dynamical systems theory suggests that when slowly varying external forcing is far from a bifurcation point2,3,9, often indicated  by a threshold value of the forcing, the state of the system remains in the basin of attraction of the quasi-static state. After any minor perturbation, the system will promptly return to its equilibrium state. As a bifurcation-induced tipping approaches, the basin of attraction undergoes a reduction in its curvature (local stability). As a consequence, even slight perturbations begin to exhibit a more prolonged effect in the dynamics, referred to as critical slowing down (CSD)1,9,13."</mark> [Page 1556](zotero://open-pdf/library/items/6PLHQUZL?page=1556&annotation=PQPYLZQV)

Explains what CSD is. Time to return to a stable state after a pertubation increases as a bifurcation comes closer. [Page 1556](zotero://open-pdf/library/items/6PLHQUZL?page=1556&annotation=PQPYLZQV)


 <mark class="hltr-green">"In the context of anthropogenic climate change, the risk of R-tipping may have been greatly underestimated, and so far, no method to anticipate such transitions has been proposed."</mark> [Page 1557](zotero://open-pdf/library/items/6PLHQUZL?page=1557&annotation=72XQILTL)

Significant gap in the literature. [Page 1557](zotero://open-pdf/library/items/6PLHQUZL?page=1557&annotation=72XQILTL)


 <mark class="hltr-purple">"The results presented in Fig. 2a align with such a scenario, and the statistics are derived from 300,000 ensemble realizations, with approximately 37% of them experiencing R-tipping. Specifically, we select 60,000 time series with R-tipping (group A) and an additional 60,000 without tipping (group B) for further analysis."</mark> [Page 1557](zotero://open-pdf/library/items/6PLHQUZL?page=1557&annotation=3YUY6JXG)


 <mark class="hltr-yellow">"Unlike bifurcation-induced tipping, R-tipping is not linked to a loss of stability of equilibrium, and the shape of the system’s basin of attraction can remain invariant; no underlying theory exists for anticipating R-tipping, such as CSD in the bifurcation-induced tipping case."</mark> [Page 1557](zotero://open-pdf/library/items/6PLHQUZL?page=1557&annotation=PKJIIIXW)

Basin of attraction remains invariant so cannot detect if a critical transition is approaching. When the attractor moves faster than [Page 1557](zotero://open-pdf/library/items/6PLHQUZL?page=1557&annotation=PKJIIIXW)


 <mark class="hltr-red">"The randomness of perturbations precludes the inference of whether a specific realization will undergo tipping based on the system’s equations alone."</mark> [Page 1557](zotero://open-pdf/library/items/6PLHQUZL?page=1557&annotation=IJ5VIREB)

Don't understand this [Page 1557](zotero://open-pdf/library/items/6PLHQUZL?page=1557&annotation=IJ5VIREB)


 <mark class="hltr-green">"Regarding the variance, the composite mean values for groups A and B display increasing trends over time, and their 99% confidence intervals substantially overlap. Hence, unlike for the case of bifurcation-induced tipping, CSD cannot discern R-tipping from the non-tipping cases amidst the changing forcing and noise perturbations. CSD can, hence, not serve to anticipate R-tipping. This may be expected since CSD focuses on changes in the linear restoring rate and how that affects the autocorrelation or variance. The underlying assumption that the system remains close to equilibrium and that the linearization around the equilibrium is valid is, by design, broken in the R-tipping context."</mark> [Page 1558](zotero://open-pdf/library/items/6PLHQUZL?page=1558&annotation=EHFAXBK7)


 <mark class="hltr-yellow">"the CSD theory relies on linearizing the dynamics around a given stable equilibrium, so using CSD indicators in cases where the system is not close to equilibrium is not mathematically justified."</mark> [Page 1558](zotero://open-pdf/library/items/6PLHQUZL?page=1558&annotation=VM287QWS)


 <mark class="hltr-green">"It is found that significant differences in probability distributions between R-tipping and non-tipping scenarios persist up to 280 time steps before the onset of R-tipping. Similar conclusions are also evident in the analyses conducted for the Bautin system and the compost-bomb system (Supplementary Figs. 1 and 2). This suggests the presence of higher-order statistical information beyond CSD, which can differentiate between R-tipping and non-tipping time series, thereby making R-tipping potentially predictable. We conjecture that these characteristic higher-order statistics represent how far the system in question is away from equilibrium. This encourages us to pursue further investigations to identify valid precursor signals for R-tipping."</mark> [Page 1558](zotero://open-pdf/library/items/6PLHQUZL?page=1558&annotation=V76LH6WZ)

CSD indicators are not significantly different between tipping and non-tipping cases. However, the probability distribution of time series values does differ for up to 280 time steps. The authors conjecture that a higher order statistic exists which could function as an early warning indicator for R-tipping. [Page 1558](zotero://open-pdf/library/items/6PLHQUZL?page=1558&annotation=V76LH6WZ)


 <mark class="hltr-yellow">"Aiming to establish a DL-based indicator for predicting R-tipping, our DL models integrate a convolutional neural network (CNN) with a fully connected neural network, enabling the extraction of both local and global information from the time series33."</mark> [Page 1558](zotero://open-pdf/library/items/6PLHQUZL?page=1558&annotation=Q6Z3PZW5)


 <mark class="hltr-green">"When feeding a time series segment into the trained DL model, the binary outputs represent the probabilities of this time series, at this lead time, to be an R-tipping scenario or a non-tipping scenario (Supplementary Fig. 3)."</mark> [Page 1558](zotero://open-pdf/library/items/6PLHQUZL?page=1558&annotation=Q97L29L2)


 <mark class="hltr-green">". Iterating predictions using the trained DL models on time series within the aforesaid groups A and B, the composite results show that this DL-derived R-tipping indicator can clearly distinguish R-tipping (group A) from non-tipping (group B) scenarios, with a long lead time before R-tipping (Fig. 3a, bottom)."</mark> [Page 1559](zotero://open-pdf/library/items/6PLHQUZL?page=1559&annotation=4R3XCDY4)


 <mark class="hltr-green">"At each forecast lead time, we employ the Kolmogorov–Smirnov significance test to assess whether the R-tipping probability of group A is distinguishable from those of group B (Supplementary Fig. 2a, bottom). The findings reveal that for up to 290 time steps before the onset of R-tipping in the saddle-node system, the DL-derived R-tipping probabilities in the two groups become distinguishable."</mark> [Page 1559](zotero://open-pdf/library/items/6PLHQUZL?page=1559&annotation=B46AWREN)


 <mark class="hltr-red">"It is noteworthy that the prediction accuracy of the DL models shows weak disparity between the training and testing phases (Supplementary Fig. 5), indicating that overfitting is not an issue for our DL model"</mark> [Page 1559](zotero://open-pdf/library/items/6PLHQUZL?page=1559&annotation=Z6Q3GANI)


 <mark class="hltr-green">"Despite different internal dynamics compared with the saddle-node system, the combined impact of rapid external forcing and noise perturbations triggering R-tipping is common to all three systems."</mark> [Page 1559](zotero://open-pdf/library/items/6PLHQUZL?page=1559&annotation=6T8LP2K9)


 <mark class="hltr-green">"Employing a consistent DL configuration and training strategy, the trained DL models, in contrast, provide reliable long-lead forecasts for R-tipping within the Bautin and compost-bomb system"</mark> [Page 1559](zotero://open-pdf/library/items/6PLHQUZL?page=1559&annotation=WJPU22WK)


 <mark class="hltr-green">"The prediction accuracy as a function of the forcing rate and forecast lead time for the saddle-node system (Fig. 5a) demonstrates that the trained DL models can well adapt to the out-of-sample forcing scenarios. With a forecast lead time of 50 time steps, the DL model exhibits higher accuracy in cases featuring lower forcing rates"</mark> [Page 1560](zotero://open-pdf/library/items/6PLHQUZL?page=1560&annotation=PJ355EKR)


 <mark class="hltr-yellow">"To fill this gap, we introduced here a skilful DL-based indicator to predict R-tipping amid noise perturbations."</mark> [Page 1561](zotero://open-pdf/library/items/6PLHQUZL?page=1561&annotation=6JH6KPRI)


 <mark class="hltr-yellow">"DL algorithm can extract high-order statistical information quantifying how far a system is away from equilibrium and, hence, how close it is to crossing the boundary of a given basin of attraction."</mark> [Page 1561](zotero://open-pdf/library/items/6PLHQUZL?page=1561&annotation=B3MIFVIV)


 <mark class="hltr-yellow">"This information can be readily used to give quantitative probabilities that an R-tipping"</mark> [Page 1561](zotero://open-pdf/library/items/6PLHQUZL?page=1561&annotation=U9AAD2B6)


 <mark class="hltr-yellow">"We prospect that a comprehensive DL model could be taken to provide precursor signals across diverse tipping phenomena and systems, as well as distinguish between them in terms of their different forcing scenarios, purely based on the data."</mark> [Page 1562](zotero://open-pdf/library/items/6PLHQUZL?page=1562&annotation=8P6DEU4U)

Big Idea [Page 1562](zotero://open-pdf/library/items/6PLHQUZL?page=1562&annotation=8P6DEU4U)




%% Import Date: 2025-10-21T12:54:53.771+01:00 %%
