# Hierarchical VAE on CIFAR-10

In this project, I derived the ELBO for both a standard VAE and a two-layer hierarchical VAE, implemented both models in PyTorch, trained them on CIFAR-10 in Google Colab, and evaluated their performance using FID scores.

## Key Ideas

**Beta Scheduling:** I used a beta schedule that ramps from 0 to 1 over the first 20 epochs, holds at 1 for 10 epochs, then decays to 0.4. The ramp-up prevents posterior collapse by letting the encoder-decoder coupling form before KL regularization kicks in. The decay relaxes the KL constraint slowly from epoch 30 onward, allowing the encoder to retain more information and improve reconstruction quality.

**Hierarchical VAE:** The standard VAE forces the encoder toward a rigid N(0, I) prior, limiting how much information the latent state can carry. The hierarchical VAE replaces this with a learned, context-dependent prior: a second latent layer z2 captures high-level structure, and a decoder produces a flexible prior for z1 conditioned on z2. This relaxes the information bottleneck, allowing for more expressive latent representations. During training, the encoder's z1 is passed directly to the decoder for reconstruction, while the learned prior only influences training through the loss function. At generation time, z1 is sampled from the learned prior since no encoder is available.

**Staggered Alpha-Beta Scheduling:** A naive hierarchical VAE with a single beta schedule underperformed the standard VAE on both metrics (reconstruction: 33,069, FID: 120.9). I observed that the middle KL term (which aligns the learned prior with the encoder) depends on z2 being well-structured, so ramping both constraints at the same rate not optimal. I designed a staggered schedule: beta controls KL on z2 with the standard schedule, while a separate alpha controls the middle term with a delayed ramp (peaking at epoch 40 instead of 20) and a lower floor (0.25 vs 0.4). This gives z2 time to organize before the prior alignment term demands that it be useful, and the lower floor gives the encoder more freedom to retain information, improving reconstruction at the cost of weaker prior alignment for generation.

## Results

|      Model       | Reconstruction Loss | FID Score |
|------------------|---------------------|-----------|
| VAE              | 30,291              | 105.5     |
| HVAE (beta only) | 33,069              | 120.9     |
| HVAE (alpha-beta)| 27,717              | 116.2     |

The staggered scheduling improved reconstruction loss by 8.5% over the single-layer VAE baseline, demonstrating that the hierarchical architecture can outperform a standard VAE when the training dynamics are managed properly. However, the FID score remained higher, which reveals a reconstruction-generation tradeoff: the lower alpha floor gave the encoder more freedom to encode information (improving reconstruction) at the cost of weaker prior alignment (hurting generation). I believe that realizing the full potential of hierarchical VAEs requires deeper architectures with convolutional latents, residual connections, and careful engineering, as demonstrated by models like NVAE and VDVAE.
