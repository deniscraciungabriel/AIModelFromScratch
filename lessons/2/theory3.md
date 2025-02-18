Weights are initialized to small random values (using a normal distribution with mean 0 and a small standard deviation like 0.02) and biases to zero for several practical reasons:

1. **Breaking Symmetry:**

   - If all weights started at the same value, every neuron in a layer would learn the same features during training.
   - Random initialization ensures that different neurons begin with different weights, allowing them to learn different patterns.

2. **Controlling the Scale of Activations:**

   - Small initial weights help keep the outputs of layers in a reasonable range.
   - If weights are too large, the activations might become too large (or saturate activation functions like ReLU or tanh), which can lead to unstable gradients during backpropagation.
   - Using a small standard deviation (like 0.02) helps maintain a stable forward and backward flow of gradients, preventing exploding or vanishing gradients.

3. **Zero Biases as a Neutral Starting Point:**

   - Initializing biases to zero is common because biases are simply offsets that the network can adjust during training.
   - Starting at zero means that, initially, the neuron's behavior is determined solely by the inputs and the weights, without any added bias.

4. **Empirical and Theoretical Best Practices:**
   - The specific initialization parameters (mean 0 and std 0.02) are often chosen based on experiments and theoretical insights in training deep networks, especially in models like GPT and other transformers.
   - These values have been found to work well in practice, leading to faster and more stable convergence.

**Example:**  
Imagine a linear layer that transforms a 384-dimensional input to a 100-dimensional output. If its weights are initialized using `normal_(mean=0, std=0.02)`, then:

- Most weight values will be very small (around zero), but slightly different from each other.
- This small randomness allows each neuron to start learning a unique mapping, while keeping the activations in check.

In summary, this initialization strategy helps the model start training in a balanced state, ensuring that gradients flow well and that the network can learn effectively from the very first training steps.
