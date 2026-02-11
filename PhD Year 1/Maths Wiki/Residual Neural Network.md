https://en.wikipedia.org/wiki/Residual_neural_network

Network attempts to learn only the residuals, rather than full output. It was first introduced in the paper (He et al., 2015) (has 290k + citations as of October 2025).

$$
x_{k+1} = x_k + f(x_k)
$$
where $f(x_k)$ is a neural network which approximates the residual

How to train in input and output have different dimensions?

Residual connections are added so original signal is not lost, not forced to use every block so original signal is not lost in a very deep network. 

**ResNet Block**
![[Drawing 2025-10-19 22.01.02.excalidraw]]