***

#### Deep Sequence Modelling

Modelling sequences is very important 

##### Recurrent Neurons

![[Mit_DL_L2_D1]]
The network need a notion of memory. 

$$\hat y_t = f(x_t, h_{t-1})$$
![[Pasted image 20251027120006.png]]

Hidden state is preserved from previous layer to function as memory. 
Example of hidden state update function

$$ h_t = tanh(W^{T}_{hh}h_{t-1} + W^{T}_{xh}x_t) $$

##### Design Criteria 

1. Handle variable-length sequences (increase to image input)
2. Track long-term dependencies.
3. Maintain information about order
4. Share parameters across the sequence. 

##### Sequence Modelling Problem: Predict the Next Word

First, a away to represent language to a neural network is needed. 

Will embed sequence into a vector space. 

**One-embedding vs Learned Embedding**

Need to identify words and their relationship to the rest of the sequence. 

##### Backpropagation Through Time 

As all steps are dependent, cannot compute loss independently on each $x$.

Have to compute gradient through each step of the sequence, if many values are greater than 1, may lead to **exploding gradients**. Similarly if many gradients are less than 1, it may vanish. This may lead to unstable training, taking steps far too large or far too small respectively. 

How can this issue be avoided?

Some these are [[Gated Recurrent Units (GRUs)]] and [[Long short term memory (LSTM)]].

##### Limitation of Recurrent Models
- Encoding bottleneck (When embedding sequence? Only certain amount can be held in memory $h_t$)
- No parallelization on a sequence. 
- Not long memory ? (from encoding bottleneck )


What if made a sequence a single put and use deep feedforward network? 
![[Pasted image 20251027124318.png]]



- Eliminate recurrence, allows for parallelization. 
- However, crucial information about sequence order is lost.
- Long term memory will be lost (from depth of network maybe?)
Can a better way be found?

##### Attention

First introduced in (Vaswani et al., 2017) (very famous paper). 

Human intuitively pick up what information is important and what it not. We pay attention to the most important elements. 

Give elements keys and compare and identity elements.  


How is this done? 

1. Encode position information through an embedding.  This removed need for recurrence, all data to shown to the network at once. 
2. Extract query, key, value. NN layer used to calculate these values. 
Query indicates what information is being searched for. 
Key indicated what is information is core to each element of the input. 
Value ?
3. Determine for what elements the query and key are similar. 
Measure similarity - both are vectors, similarity taken by dot product. 

Determine what parts of the input are more important what elements are relevant to each other. Use softmax function to get probability. 

How much attention does each element of the input place on other element of the input? 

Does each element of the input have its own query and key? Then matrix of attention weighting is calculated for the input. 

Value is original input, it is scaled by attention distribution derived by the query and key. 
**How can this be scaled to an actual ML architecture?**

![[Pasted image 20251027130854.png]]![[Pasted image 20251027131219.png]]
