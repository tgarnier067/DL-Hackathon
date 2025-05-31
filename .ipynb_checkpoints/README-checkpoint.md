# Objective

This project participates in a hackathon : **Learning with noisy graphs labels**.

---

# Dataset

The dataset originates from the publicly accessible Protein-Protein Association (PPA) dataset, specifically the ogbg-ppa version. This dataset includes 37 classes.

For this study: A random selection of 6 classes was made from the original 37. Additionally, 40% of the dataset was randomly sampled.

Noise Introduction: Various degrees of symmetric and asymmetric noise were injected into the labels. As a result, four separate datasets were generated, each differing by: The proportion of noisy labels, or the type of noise applied (symmetric versus asymmetric).

The dataset can be downloaded [here](https://drive.google.com/drive/folders/1Z-1JkPJ6q4C6jX4brvq1VRbJH5RPUCAk?usp=drive_link)

---

# Methods

We employed a custom loss function called **Weighted Cross Entropy**, which builds upon the standard cross-entropy loss by weighting each sample’s loss according to the predicted probability of the true class. This weighting is controlled by a parameter gamma (set to 0.2), which modulates the influence of confident predictions, allowing the model to focus more on harder or uncertain examples during training. This approach helps improve robustness against noisy labels.We applied this loss function specifically on **Graph Neural Networks** (GNNs) to enhance their learning capacity in noisy label conditions.

We also experimented with more advanced methods such as GCOD (see the reference part), which are designed to handle noisy labels more effectively. However, due to limited computational resources, **we were unable to train these models for a sufficient number of epochs to fully leverage their potential.**

In addition to our primary approach, we explored a wide range of loss functions and training algorithms to improve model robustness against label noise. The losses we tested include Cross Entropy (CE), Noisy CE, Generalized CE, Symmetric CE, Forward Correction for Categorical CE, as well as NCOD, NCOD+, and GCOD. On the algorithmic side, we experimented with advanced graph neural network architectures such as Graph Attention Networks (GAT), and techniques like Early Learning Regularization, DivideMix, and Co-teaching to further enhance performance.

All the secondary approachs where exploired thanks to the notebooks in the folder `experiments`

---

# Project structure

- `main.py` : main script for training and inference
- `source/` : source code
- `experiments/` : notebooks for experimenting with secondary methods
- `checkpoints/` : saved models
- `logs/` : plots of train and validation, loss and accuracy
- `submission/` : prediction results


---

# Prediction

To make predictions on the trained models :

```bash
python main.py --test_path ../A/test.json.gz
```

To train a model and make predictions : 
```bash
python main.py --test_path ../A/test.json.gz --train_path ../A/train.json.gz
```

---

# Results


Our method effectively learns from data even in the presence of noisy labels, demonstrating robustness and stable performance across different noise levels. The custom Weighted Cross Entropy loss helps the model focus on harder examples, improving its ability to generalize despite label noise.

More advanced techniques like GCOD have shown promising results and are expected to yield even better performance. However, fully leveraging these methods requires significant computational resources and longer training times, which were beyond the scope of this project due to hardware limitations.

---

# Reference

- [wani et al. (2024)](https://arxiv.org/abs/2412.08419) : Robustness of Graph Classification: failure modes, causes, and noise-resistant loss in Graph Neural Networks

- [Wani and al. (2023)](https://arxiv.org/abs/2303.09470) : Learning with Noisy Labels through Learnable Weighting and Centroid Similarity

---

# Contact

For any question, please contact Thomas Garnier :  **tgarnier067@gmail.com**




