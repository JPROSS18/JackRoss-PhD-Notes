


#### Categories

Saddle-node
Pitchfork
Trans critical 



Homoclinic
Heteroclinic

## Hopf Bifurcation 

$$
\begin{aligned}
\dot r = r (\rho + \alpha r^2) \\
\dot \theta = \omega + \beta r^2
\end{aligned}
$$

$\dot r = 0$ when $r = 0, \sqrt{-\frac{\rho}{\alpha}}$

**In Cartesian co-ords:** 

$$
\begin{align}
\dot x = \rho x - \omega y + (\alpha x - \beta y)(x^2 + y^2) \\
\dot y = \omega x + \rho y + (\beta x + \alpha y)(x^2 + y^2)
\end{align}
$$

Which is linearised as:
$$
\begin{align}
\dot x = \rho x - \omega y \\
\dot y = \omega x + \rho y 
\end{align}
$$
Jacobian:


$$ J = \begin{pmatrix} \rho & -1 \\ 1 & \rho \end{pmatrix} $$
Eigenvalues:
$$
\lambda_{1,2} = \rho \pm \omega i
$$


The hopf bifurcation can be divided into supercritical ($\alpha < 0$ stable limit cycle) and subcritical ($\alpha > 0$ ). Similar to the pitchfork bifurcation 

**Subcritical ($\alpha$ >0):**



**Supercritical ($\alpha < 0$):**



