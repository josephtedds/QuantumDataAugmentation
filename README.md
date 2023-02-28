# Quantum Data Augmentation

This provides an overview of the repository. If you want to explore any of these ideas further, then see [quick_start.ipynb](data_augmentation/quick_start.ipynb).

 1. **Motivation** - interesting algorithms in the NISQ era have low resource costs and are robust to noise.
 2. **Data augmentation** - classical and quantum data augmentation techniques.
 3. **Hyperparameter optimisation** - experimenting with larger and stronger blurring .
 4. **Mixup** - a novel quantum method for data augmentation.

I provide a resource estimator for the number of circuit executions and time it takes to run VQC-based machine learning methods and show it suggests we need to think more carefully about applications of QML. Motivated by this, I use Wooton et al.'s [Quantum Blur](https://arxiv.org/abs/2112.01646) as a data augmentation technique for the CIFAR-10 dataset, achieving human-level performance using David Page's [ResNet model](https://myrtle.ai/learn/how-to-train-your-resnet/) and demonstrates that approach has merit for further study. Finally, I develop a novel Quantum Mixup method for data augmentation, though analysis of this is left to further study.

# Motivation

We are currently in the so-called Noisy-Intermediate Scale Quantum (NISQ) era where we have relatively low numbers of qubits and large amounts of noise. Many of the algorithms we get exponential speedups from for instance, Quantum Phase Estimation, require error correction and cannot be feasibly implemented at this time. We're interested therefore, if there are any algorithms we can currently implement that provide value for quantum computing.

## Quantum Machine Learning

Quantum Machine Learning (QML) is one such area containing algorithms that could be of interest in the NISQ era. Roughly speaking, QML can be described as any machine learning approach that makes use of a quantum computer. While this includes techniques that are closer to classical machine learning, for example [k-means clustering](https://arxiv.org/abs/1812.03584) or [quantum neural networks](https://arxiv.org/abs/2011.00027), this also encompasses [classical machine learning on quantum data](https://arxiv.org/abs/2002.08953).

Many of these algorithms however, are not implementable on current quantum devices. The requirement for an efficient state preparation routine, or the use of QRAM renders these unusable for some time to come. Even if it these problems didn't occur, the circuits required to run these algorithms are far deeper than we're currently capable of, and so the the results are dominated by hardware noise.

One of the favoured approaches that can be implemented on current machines is to to use Parameterised or Variational Quantum Circuits (PQCs or VQCs). Heuristically, the hope is that the parameters of the circuits being trainable means that the circuit is allowed to learn and adapt to the machine noise, mitigating some of the noise problems on near-term devices. 

VQCs typically consist of 3 distinct parts:
 - An embedding for classical data, often angle embedding.
 - A trainable quantum circuit, where the ansatz is picked from a standard set, or designed to increase performance on a given dataset.
 - Measurement of the quantum state.
 
<p align="center">
  <img src="figures\README_imgs\vqc.png" alt="VQC structure"/>
</p>

This can be combined with larger classical networks that reduce the number of features in a dataset to a size suitable for embedding on a quantum computer. Then, this model is paired with a loss function and optimiser to train.  

Difficulty arises from the fact that seemingly simple circuits need thousands of circuit executions to train. Let's work through some examples and see.

At a basic level, we need to consider:
 - Training set size
 - Number of epochs run
 - Parameterised circuit structure
 - [Method for gradient calculation](https://docs.pennylane.ai/en/stable/introduction/interfaces.html)
 - Whether there are trainable layers before the quantum circuit, and if so, the parameterisation of the classical data embedding

Extensions of this should also take into account [multi-programming](https://memlab.ece.gatech.edu/papers/MICRO_2019_4.pdf), error mitigation or error suppression techniques and batching interactions.

For the following, we take the numbered ansatze from [Sim et al.](https://arxiv.org/abs/1905.10876) on 4 qubits, with a depth of 1 and no trainable layers before the quantum circuit. 

|Circuit ansatz  |Gradient method|Training set size  | Epochs |Number of Circuits |
|--|--|:--:|:--:|:--:|
|Sim2|Finite Difference  | 1000 |100  |900,000
|Sim2|Parameter Shift  | 1000 | 100 |1,700,000
|Sim18|Finite Difference  | 1000 | 100 |1,300,000
|Sim18|Parameter Shift  | 1000 | 100 |3,300,000
|Fixed circuit | N/A | 50,000 | 20 | 1,000,000

Note that all the above choices can be customised in [ml_training.py](data_augmentation/ml_training.py).

The number of circuit executions here are enormous. If the models take 0.1 seconds to run a single circuit (including all the classical overhead for loading, offloading, compilation and transpilation), then the first model would take just over a day's worth of QPU time to run. Realistically, this is likely to be infeasible unless you're one of the few people to own a quantum computer. Worse, these circuits aren't particularly complex or deep.
Consider:

 - Sim7 ansatz on 10 qubits with 4 layers.
 - Parameter shift gradients
 - 100,000 training data points for 20 epochs
 - No pre-quantum layers

This requires 930,000,000 circuits to complete, somewhere in the region of 2-3 years with 0.1 seconds per circuit.

There are some mitigations we can put in place, for instance the use of multi-programming i.e. stacking circuits next to each other so they can run on a larger machine. However, this may require fundamentally changing how we view either the use of quantum for QML or the methods for resource reduction.

If we return to the above table, we can see that a fixed circuit has many less circuit execution than some of these simple models - and this still holds even if the complexity of the circuit increases. If we can make use of this in some machine learning process, where the noise of the quantum circuit is helpful, this could make a good candidate for further exploration. 

# Data augmentation
Data augmentation in classical computing is a technique to increase the performance of models without needing to collect more data. Instead, existing data is perturbed or transformed in some fashion to create similar but non-identical data. This can take place statically, by generating extra data before the beginning of a machine learning process or dynamically as each piece of data is loaded. This can be used to help encode biases of the data or make the model more robust to noise. For classification problems, in its simplest form, the transformations of the data map classes to the same class, a dog to a dog, a cat to a cat. 

 - Translation - padding an image and randomly cropping it translates the image. This encodes a translation-invariant bias: a cat is a cat whether it's in the top-right or the bottom-left.
 - Mirroring - images can be flipped left to right. A cat in a mirror is still a cat.
 - Cutout - sections of the image are completely cutout and replaced by blocks of colour. This increases robustness to noise.
 - Gaussian noise - sections of the image have Gaussian noise added to it (iid normal distributions). This increases robustness to noise.

<p align="center">
  <img src="figures\README_imgs\classical_data_aug_small.png" alt="Classical data augmentation techniques"/>
</p>

Given that quantum hardware is inherently noisy, adding some noise via a fixed or random circuit seems like something that should be possible especially in the near-term. Noise in this case can be [treated as a resource](https://arxiv.org/abs/2302.06584) for us.

So, how do we convert noise on quantum hardware to quantum data augmentation techniques? We need an encoding of images that is efficient to both encode and decode. A simple choice in this instance is to use angle embedding with each qubit corresponding to a single pixel, and Z-measurements corresponding to the relative brightness of a pixel. Allowing the circuit to evolve for a period of time without interacting would apply the quantum hardware noise directly to the image.

We'll take an alternative approach and make use of [Quantum Blur](https://arxiv.org/abs/2112.01646), a technique for both encoding / decoding image data into a quantum state and adding noise to it. For a given image, the grey-scale values (integers in the range [0, 255) ) are mapped to a height value $h(x,y)$ for each pixel labelled by $(x,y)$. Combining this with a mapping  to binary strings, $s : (x,y) \mapsto b \in \{0,1\}^n$, we can form a quantum state,
$$
|h \rangle = \frac{\sum_{(x,y)} \sqrt{h(x,y)} | s(x,y)\rangle}{\sqrt{\sum_{(x,y)} h(x,y)}}.
$$

For an RGB image, you obtain three quantum states $|h_R \rangle, |h_G \rangle, |h_B \rangle$ corresponding to the height maps for each RGB component. For decoding, the magnitude of the amplitude is unknown, so the largest value is mapped to 255. In practice, this arbitrary rescaling could be replaced with a high percentile value of the height distribution of the test set so it's more representative of the original image.

Now, the choice of map $s(x,y)$ is such that the Hamming distance of neighbouring pixels is equal to 1. This means that interacting with a given qubit will also interact with neighbouring pixels (and some longer range correlations due to the mapping construction). Manipulating this embedding is achieved by applying rotational gates, which for an RX gate with small angle $\theta$ transforms  
$$\sqrt{h(x,y)} \to i \sqrt{h(x,y)} + \frac{\theta}{2}\sum_j \sqrt{h(x_j, y_j)},$$
 for pixels $(x_j, y_j)$ that have mappings $s(x_j, y_j)$ differing from $s(x,y)$ by only one bit. So for small angles, this is close to linearly interpolating between 'close' pixels. 

<p align="center">
  <img src="figures\README_imgs\quantum_blur_sizes.png" alt="Quantum blurs with different patch sizes"/>
</p>

The above image shows quantum blur with different patch sizes for the image, and acting on it with $\text{RX}(\alpha),$ with $\alpha = 0$.

For the experiments carried out below, we'll be using the [res-net](https://arxiv.org/abs/1512.03385) model from [myrtle.ai](https://myrtle.ai/learn/how-to-train-your-resnet/) as it's ridiculously fast to run with classical data augmentation. The original model runs for 24 epochs and on average achieves the human level accuracy of 94% on a 10 class classification problem. The dataset is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) a collection of 32 x 32 colour images in 10 classes (a mixture of animals and vehicles)  with a training set of size 50,000 images and 10,000 images in the test set. 

First, we'll compare this model to classical data augmentations to understand if quantum data augmentation can prove useful.

<p align="center">
  <img src="figures\training_performance_run_0_train_loss.png" alt="Loss on the training set for different data augmentation techniques"/>
</p>

For the experiment, we ran the ResNet model for 20 epochs, normalising the data from training set metrics then applying the translation (padding + cropping) and mirroring transformations before a final data augmentation technique. This last technique is one of:
 - Identity (do nothing)
 - Cutout
 - Gaussian blur, with location 0 and scale 0.01 (the graphs below suggest the choice of scale parameter was too small to have meaningful impact)
 - Quantum blur, with $\alpha=0.1$
Where the patches are 8x8 and chosen at random every epoch. Each model ran 3 times, and one of these is shown here (the other data can be found in the *experiment_runs* folder and these values seem to be fairly stable).

As the graph above shows, the separations in training loss emerged early on and remained until the end. While the validation losses were less stable, the fact that all of the validation losses (and accuracies) were near-identical at the end of the training suggests a more difficult dataset or simpler model will be needed to evaluate the overall performance of quantum blur. The main constraint on this method was the ability to run the quantum simulations. As a first implementation, it was about 40x slower to run the quantum blur compared to the cutout layer. However, the fact that this method can run on a large dataset and successfully train a model is encouraging as a first pass.

<p align="center">
  <img src="figures\training_performance_run_0_valid_loss.png" alt="Loss on the test set for different data augmentation techniques"/>
</p>

Note that if we can efficiently implement these transformations, then this may be suitable for use primarily as quantum-inspired approach, where larger patch sizes are broken down into smaller patches to reamin simulatable.

# Hyperparameter optimisation

In addition, a random hyperparameter optimisation sweep was run to determine if there were dramatic changes in performance of the model with different sized patches or different intensities of blur. This sweep ran identically to the earlier experiments: 20 epochs on the CIFAR-10 dataset, with 3 models per set of hyperparameters.

As the graphs below show, there isn't much difference in the performance for the differnt models and as above, a more sensible comparison - either with a simpler model or more difficult dataset is likely required to show separation between augmentations.

<p align="center">
  <img src="figures\hyperparam_opt_run0_train_loss.png" alt="Training set loss of hyperparameter optimisation run for quantum blur"/>
</p>
<p align="center">
  <img src="figures\hyperparam_opt_run0_valid_loss.png" alt="Test set loss of hyperparameter optimisation run for quantum blur"/>
</p>

# Mixup
[Mixup](https://arxiv.org/abs/1710.09412) is another classical data augmentation technique. Here, two images are combined together linearly to create an image part-way between two images, and the corresponding labels are also linearly combined.

Explicitly, for data taking the form of image-class pairs $(x, y), \, x \in \mathbb{R}^{3 \times n \times m}, \, y \in \{0,1\}^2$, we can define the mixup of two images as the pair $(x^\ast, y^\ast)$ where
$$
x^\ast = \lambda x_0 + (1-\lambda )x_1, \quad y^\ast = \lambda y_0 + (1-\lambda )y_1, \quad \text{for } \lambda \in (0,1).
$$
This requires us to use one-hot encodings of class labels, so each input data is a binary vector instead of a class number so we can linearly interpolate the labels and it naturally extends to beyond binary classification. This transformation occurs once a batch of images has been chosen to process and does not require that the images are from different classes.

In quantum computing, we can also combine states in a sensible fashion by considering the SWAP gate. The action of the SWAP gate is given by 
$$
\text{SWAP} | x_0\rangle | x_1\rangle = |x_1\rangle | x_0 \rangle. 
$$
As the SWAP gate is a unitary matrix, we can also take fractional powers of the SWAP gate to [combine images](https://arxiv.org/abs/2007.11510). This provides us with a Quantum Mixup method, also parameterised by $\lambda \in (0,1)$ for gate application $\text{SWAP}^\lambda$.  It is not immediately clear what label value the new image should take. While linear interpolation, as in the classical case, could be sensible, a shifted and rescaled $\tanh$ function could also provide sensible labelling. 

Due to time constraints, we have been unable to explore but provide comparisons to classical mixup on two CIFAR-10 images.


# Conclusions
As with many QML experiments, we are unable to conclude much more than the method is successful at achieving the task. However, given the size of the dataset trained on, this is encouraging, as it allows studies of dataset sizes that may be closer to real-world applications.

It also introduces a number of opportunities to explore this further:

 - Expansion of the resource estimation to account for additional strategies like multi-programming, different classical embeddings and batch-sizes. 
 - Integration of the resource estimation with PennyLane differentiation transforms to analyse arbitrary quantum circuits.
 - Running the model on quantum hardware to understand the true practicality of the approach.
 - Alternative embeddings for images such as angle embedding.
 - Training with Quantum Mixup to determine whether this approach has merit.

Hopefully this leaves readers better aware of the practical constraints of QML and encourages thinking about alternative approaches for the NISQ era.

<p>
  <img src="figures\README_imgs\mixup_comparison.png" alt="Mixup strategies on CIFAR-10"/>
</p>

For easy comparison we have inverted the relationship with $\lambda$ for the Quantum Mixup. At $\lambda = 0$ this coincides with the identity on the first image, but for classical mixup this weights the linear combination entirely in favour of the second image. From the above image, it's clear that the quantum image is far less meaningful at $\lambda = 0.5$ and more blurry throughout the process. This suggests we may need to be careful about which values of $\lambda$ we sample from so we don't feed too much noise to the ML model. Additionally, the choice of label for each quantum mixup image is unlikely to be as straightforward as a linear combination of labels as with classical mixup. Instead, the suggested choice of a rescaled and recentered $\tanh$ function may be more suited. It is worth noting that this implementation is far slower than the quantum blur, and substantial speedup would be required to attack the CIFAR-10 dataset with this.

## Acknowledgements
Thanks must go to James R. Wootton and Marcel Pfaffhauser for their experiments with and implementation Quantum Blur. Additionally to David Page for a lightning-fast model for classification on CIFAR-10.

And lastly: 
> Written with [StackEdit](https://stackedit.io/).
