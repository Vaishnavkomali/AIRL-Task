# AIRL-Task

# AIRL Internship Coding Assignment Submission

This repository contains the solutions for the two required tasks: Q1 (ViT on CIFAR-10) and Q2 (Text-Driven Segmentation).

**Setup Instructions:**
Both `q1.ipynb` and `q2.ipynb` must be executed end-to-end on Google Colab with a **GPU runtime enabled**. Ensure all installation cells at the top of each notebook are run first.

---

### 2. Q1 Vision Transformer (ViT) on CIFAR-10

#### Model Configuration for Best Result

Our objective was to achieve the highest possible test accuracy[cite: 11]. The final model configuration, which delivered the best results, incorporated several optimization tricks:

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Backbone** | ViT-Tiny (for fast training) | Utilized a compact ViT architecture. |
| **Epochs** | 150 | Full training run. |
| **Patch Size** | 4x4 | Smaller patches provide higher resolution features for CIFAR-10's 32x32 images. |
| **Optimizer** | AdamW | Standard choice for Transformer models. |
| **Scheduler** | Cosine Annealing + Warmup | Used 10 epochs of linear warmup followed by cosine decay for stable convergence. |
| **Augmentations** | CutMix / Random Cropping / Horizontal Flip | Strong regularization to prevent overfitting on CIFAR-10. |

#### Results Table

| Metric | Overall Classification Test Accuracy |
| :--- | :--- |
| **Best Accuracy** | **83.78%** |

#### Bonus Analysis: Augmentation Effect



The introduction of **CutMix** was crucial. CutMix acts as a powerful regularization technique by creating training samples composed of two images and their labels. This regularization effect stabilized the training and prevented the early convergence that was observed when only using standard Random Cropping and Flipping. The initial ViT model quickly overfit the small CIFAR-10 dataset without these advanced augmentations, demonstrating that data augmentation is more critical than architecture scaling for this small-image task.

---

### 3. Q2 Text-Driven Image Segmentation

#### Pipeline Description

The text-driven segmentation process requires converting a text query into a precise mask. Due to stability issues with other grounding models, we utilized a two-stage pipeline for robust, zero-shot segmentation:

1.  **Text-to-Seed (CLIPSeg):** The **CLIPSeg** model first processes the image and text prompt to generate a low-resolution probability map (or mask) of the target object.
2.  **Seed Conversion:** The minimum bounding box around this high-probability region is calculated. This box serves as the mandatory **region seed**.
3.  **Mask Refinement (SAM):** The bounding box seed is then passed to the **Segment Anything Model (SAM)**, which uses the geometric cue to refine the mask, generating a highly accurate, pixel-level segmentation of the object.

#### Limitations

1.  **Dependency on Grounding:** The final mask quality is fully dependent on the **CLIPSeg** model successfully locating the object and generating a good initial mask. If the text prompt is ambiguous or the object is heavily occluded, the initial seed will be poor, leading to a poor final SAM mask.
2. **Computational Cost:** The entire pipeline requires significant computational resources, primarily due to the large transformer backbones in both CLIPSeg and SAM, necessitating a GPU for timely execution.
3.  **Network Reliance:** The failure to run more examples was due to persistent Colab network restrictions on downloading external image URLs (e.g., `403 Forbidden` errors), demonstrating a practical limitation of relying on remote data fetching in dynamic cloud environments.
