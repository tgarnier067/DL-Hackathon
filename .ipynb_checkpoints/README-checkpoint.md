# DL-Hackathon

## Objective

This project participates in a hackathon on graphs classification with noise. The goal is get the higher F1-score averaged over the 4 datasets.

## Dataset

The dataset originates from the publicly accessible Protein-Protein Association (PPA) dataset, specifically the ogbg-ppa version. This dataset includes 37 classes.

For this study: A random selection of 6 classes was made from the original 37. Additionally, 40% of the dataset was randomly sampled.

Noise Introduction: Various degrees of symmetric and asymmetric noise were injected into the labels. As a result, four separate datasets were generated, each differing by: The proportion of noisy labels, or the type of noise applied (symmetric versus asymmetric).

The dataset can be downloaded [here](https://drive.google.com/drive/folders/1Z-1JkPJ6q4C6jX4brvq1VRbJH5RPUCAk?usp=drive_link)

---

## Methods

We also experimented with many other approaches (see `Many notebooks` folder), including:

We employed a custom loss function called **Weighted Cross Entropy**, which builds upon the standard cross-entropy loss by weighting each sample’s loss according to the predicted probability of the true class. This weighting is controlled by a parameter gamma (set to 0.2), which modulates the influence of confident predictions, allowing the model to focus more on harder or uncertain examples during training. This approach helps improve robustness against noisy labels.

We also experimented with more advanced methods such as GCOD (see [wani et al. (2024)](https://arxiv.org/abs/2412.08419) and [Wani and al. (2023)](https://arxiv.org/abs/2303.09470)), which are designed to handle noisy labels more effectively. However, due to limited computational resources, **we were unable to train these models for a sufficient number of epochs to fully leverage their potential.**

In addition to our primary approach, we explored a wide range of loss functions and training algorithms to improve model robustness against label noise. The losses we tested include Cross Entropy (CE), Noisy CE, Generalized CE, Symmetric CE, Forward Correction for Categorical CE, as well as NCOD, NCOD+, and GCOD. On the algorithmic side, we experimented with advanced graph neural network architectures such as Graph Attention Networks (GAT), and techniques like Early Learning Regularization, DivideMix, and Co-teaching to further enhance performance.

---

## Project Structure

