# Self-Explaining MRI GAN

## Overview

This project implements a Generative Adversarial Network (GAN) trained on brain MRI images and augments it with Grad-CAM (Gradient-weighted Class Activation Mapping) interpretability. 

The project visualizes which anatomical regions influence the discriminator’s realism judgments, enabling transparent and interpretable generative modeling in a medical imaging context.

*Areas Focused*
- Explainable AI                                         explains how discriminator thinks a generated image is real
- Generative Adversarial Networks (GANs)                 learns to generate MRI images 
- Computer Vision                                        for image realism assessment       


## Dataset

- Can be downloaded from [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Brain MRI images organized into class-specific folders
- Images are treated as **unlabeled** for unconditional GAN training
- All images are:
  - Converted to grayscale
  - Resized to a fixed spatial resolution
  - Normalized to the range `[-1, 1]`

The dataset is used solely for learning the global distribution of MRI appearance, not for classification or diagnosis.

---

## Model Architecture

### Generator

The generator model maps random noise vectors to synthetic MRI images.

- Input: Random noise vector sampled from a standard normal distribution
- Architecture:
  - Series of transposed convolution layers
  - Progressive spatial upsampling
  - Batch normalization and ReLU activations
- Output:
  - Single-channel MRI image
  - `tanh` activation to match data normalization

The generator learns to synthesize images that resemble the global structure and texture of real brain MRIs.

---

### Discriminator

The discriminator model compares real MRI images (from dataset) with generated MRI images (from generator) and outputs a scalar realism score (logit).

- Input: Real or generated MRI image
- Architecture:
  - Hierarchical convolutional feature extractor
  - Progressive spatial downsampling
  - LeakyReLU activations
- Output:
  - Scalar realism score (logit)
  - Intermediate feature maps for interpretability

The discriminator learns to distinguish real MRI images from generated ones by identifying structural and textural cues.

---

## Training Procedure

The generator and discriminator are trained adversarially using *binary cross-entropy loss with logits* (inclusive of the sigmoid activation and numerically stable)

- The discriminator learns to:
  - Classify real images as real
  - Classify generated images as fake
- The generator learns to:
  - Fool the discriminator into classifying generated images as real

Training is monitored through:
- Stable oscillatory loss behavior
- Qualitative inspection of generated samples

The model is trained once using GPU acceleration (provided by Google Colab) and the learned weights are saved for later analysis and visualization.

---

## Results: Image Generation

After training, the generator produces MRI-like images that exhibit:
- Coherent skull boundaries
- Realistic brain shapes
- Plausible intensity distributions
- Diversity across samples

These images are not replicas of the training data but novel samples drawn from the learned distribution.

---

## Interpretability with Grad-CAM

To analyze how the discriminator evaluates realism, Grad-CAM is applied to its convolutional feature maps.

Grad-CAM (Gradient-weighted Class Activation Mapping) is a visualization technique that uses gradients to highlight the image regions that most influence a neural network’s decision.

### Observations

- For **generated MRI images**, Grad-CAM highlights:
  - Skull boundaries
  - Central brain regions
  - Structural texture patterns
- For **real MRI images**, Grad-CAM highlights:
  - Similar anatomical regions
  - Comparable global structures

- **Realism is based on anatomical features rather than spurious artifacts**.

---

## Key Insights

- The GAN successfully *learns global MRI anatomy rather than memorizing noise*
- Discriminator attention aligns with medically meaningful structures
- Grad-CAM provides a transparent view into the discriminator’s decision process
- Interpretability can be integrated into generative models without altering training

---

## Limitations

- Generated images are not intended for clinical diagnosis
- The model is trained on unconditional GAN
- Grad-CAM provides qualitative explanations rather than formal guarantees

---

## Conclusion

This project demonstrates that GAN-based MRI image synthesis can be augmented with visual interpretability techniques to better understand model behavior.

---

## Disclaimer

Not for Clinical Decision-Making, Solely for Educational and Research Purposes

Worked by

Shajil BP
