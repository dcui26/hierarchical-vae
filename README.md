# Hierarchical VAE on CIFAR-10

In this project, I derived the ELBO for both a standard VAE and a two-layer hierarchical VAE, implemented both models in PyTorch, trained them on CIFAR-10 in Google Colab, and evaluated their performance using FID scores.

# Key Ideas

Beta Scheduling: I used a beta schedule that ramps from 0 to 1 over the first 20 epochs, holds at 1 for 10 epochs, then decays to 0.4.
The ramp-up prevents posterior collapse by letting the encoder-decoder coupling form before KL regularization kicks in.
The decay relaxes the KL constraint in slowly from 30 epochs to the end, allowing the encoder to retain more information and improve reconstruction quality.

Hierarchical VAE: The standard VAE forces the encoder toward a rigid N(0, I) prior, limiting how much information the latent code can carry.
The hierarchical VAE replaces this with a learned, context-dependent prior: a second latent layer z2 captures high-level structure, and a decoder produces a flexible prior for z1 conditioned on z2.
This relaxes the information bottleneck in theory, allowing for more expressive information retention.

# Results

| Model | Reconstruction Loss | FID Score |
|-------|---------------------|-----------|
| VAE   | 30,291              | 105.5     |
| HVAE  | 33,069              | 120.9     |

The single-layer VAE outperformed the hierarchical VAE on both metrics.
I speculate that the theoretical advantage of flexible learned priors requires deeper architectures with more complex design choices (convolutional latents, residual connections, 30+ layers as in NVAE/VDVAE) to be obtained.
At this scale of my project, the optimization complexity of coordinating four networks outweighs the benefit of the relaxed information bottleneck.
