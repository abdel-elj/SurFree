# SurFree Attack Walkthrough

This repository provides a step-by-step guide to set up and run the **SurFree** attack – a surrogate-free black-box adversarial attack based on Foolbox.

📄 **Original Paper**: [SurFree: a fast surrogate-free black-box attack](https://arxiv.org/abs/2011.12807)  
🔗 **Original GitHub**: [t-maho/SurFree](https://github.com/t-maho/SurFree)

---

## 📌 About SurFree

SurFree is a **black-box decision-based attack** that generates adversarial examples with minimal queries. Unlike other attacks (e.g., HSJA, QEBA, GeoDA), SurFree does **not rely on gradient estimation**. Instead, it explores geometrical directions to cross the decision boundary efficiently.

The key idea is to bypass gradient estimation entirely and instead focus on careful, geometrically-guided trials along diverse directions. By avoiding the query-intensive process of estimating gradients, SurFree aims to achieve a significant reduction in the number of queries required to find adversarial examples, particularly under low query budgets (hundreds to a thousand queries). It iteratively refines an adversarial example by searching along orthogonal directions based on geometric properties of hyperplane decision boundaries, aiming for rapid distortion reduction.

## SurFree Algorithm

The core SurFree algorithm operates iteratively. Here's a high-level overview of the steps involved in one iteration (refer to Algorithm 1 in the paper for details):

1.  **Initialization:** Start with an adversarial example x_{b,k} close to the decision boundary. (The very first x_{b,1} is found using an initial adversarial point and a binary search towards the original image x_o).
2.  **New Direction:**
    * Define the current direction u_k from x_o to x_{b,k}.
    * Generate a pseudo-random direction t_k (often using DCT, see Section 5.2).
    * Orthogonalize t_k against u_k and recent previous directions to get the search direction v_k.
3.  **Sign Search:**
    * Test points along a circular path defined by u_k and v_k, starting with angles \pm \theta_{max} and decreasing magnitude.
    * Query the model for each test point.
    * Stop if an adversarial point is found. If no adversarial point is found for direction v_k, reduce \theta_{max} slightly and go back to Step 2 to generate a new direction.
4.  **Binary Search:**
    * If the Sign Search found an adversarial point at angle \theta_{test}, perform a binary search in the angular interval between the last non-adversarial angle and \theta_{test}.
    * Refine the angle to find \theta^* corresponding to a point x_{b,k+1} = z^*(\theta^*) very close to the boundary.
5.  **Output:** Return the new boundary point x_{b,k+1} for the next iteration.
6.  **(Optional) Interpolation:** An optional step (Section 5.3) can be used after the Binary Search to potentially find an even closer point by modeling boundary curvature using an additional query.

---
### My Contribution
#### Targeted-Attack Surfree
This version of the SurFree algorithm is adapted for targeted adversarial attacks, where the goal is to perturb an input sample such that it is misclassified as a specific target class rather than just any incorrect label.
1.  **Initialization:** Start with an adversarial example x_{b,1} that is classified as the **target class** t. This is done by generating an initial adversarial candidate close to the decision boundary and performing a binary search between the original input x_o and the initial candidate until a point classified as t is found.

2.  **New Direction:**
    * Define the current direction u_k from x_o to x_{b,k}.
    * Generate a pseudo-random direction t_k (e.g., using DCT).
    * Orthogonalize t_k against u_k and recent previous directions to get the new search direction v_k.

3.  **Sign Search (Targeted):**
    * Generate test points along a circular arc in the plane of u_k and v_k, using angles \pm \theta_{max}.
    * For each test point z(\theta), query the model.
    * Check if the prediction equals the **target class** t.
    * If no such target-class point is found, reduce \theta_{max} slightly and return to Step 2 with a new direction.

4.  **Binary Search (Targeted):**
    * If a point z(\theta_{test}) classified as the target class is found during Sign Search, perform a binary search along the angular path from the original input toward z(\theta_{test}).
    * Refine the angle to find the point x_{b,k+1} = z^*(\theta^*) that lies on the boundary and is classified as the **target class**.

5.  **Output:** Return the new adversarial boundary point x_{b,k+1} that is as close as possible to x_o and is confidently classified as the target class t.

#### Enhanced Initialization (Experiment)
**This will not result in lower queries, instead of that, it results in lower distortion using the same number of steps**

Giving the algorithm a push in the initialization phase in order to reach minimum distances with the same steps number.​

- For each sample in X, generate n_directions orthogonal to it, apply perturbations, and select the best adversarial candidate (i.e., one that fools the model with minimal L2 distance).​

## ⚙️ Installation & Setup

Follow these steps to prepare the environment and run the attack.

### 1. Clone the Repositories

```bash
# Clone the SurFree attack
git clone https://github.com/t-maho/SurFree.git
cd SurFree
```

## 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate
```
## 3. Install dependencies:
```bash
pip install -r requirements.txt 
```

## 4. Run the attack:
### Untargeted Attack
```bash
run main.py
```
### Targeted Attack
```bash
run main_targeted.py
```
**NOTE: I configured the code to run on only one image and for targeted attack with target class (577), rest of image sample located in back_images folder, just move them to images folder to run the attack with the other images**
- ```Change the target label from line 98 in surfree_targeted.py```
- ``` config_example.json contains all parameters configuration```
## Models

The experiments in the paper were conducted using the following models:

* **For MNIST**, they use a pre-trained CNN network that is composed of 2 convolutional layers and 2 fully connected Layers.
* **ImageNet:** The ImageNet dataset is tackled by a pre-trained ResNet18, made available for the PyTorch environment.

## Requirements
```
torch==2.6.0
eagerpy==0.30.0
scipy==1.15.2
tqdm==4.67.1
scikit-learn==1.6.1
GitPython==3.1.44
requests==2.32.3
torchvision==0.21.0
```
## Contact

For future work or enhancement contact at:
* Abdelrahman ElJamal: `abdelrahman.eljamal@uri.edu`
