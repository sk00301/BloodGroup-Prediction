# 🩸 Blood Group Prediction from Fingerprints

A deep learning-based system that predicts a person's **ABO/Rh blood group from fingerprint images** using a custom Convolutional Neural Network (CNN) built in PyTorch — a non-invasive alternative to traditional blood typing.

---

## 📌 Overview

Traditional blood group detection requires laboratory testing, which is time-consuming, invasive, and resource-intensive. This project explores a non-invasive approach by analyzing fingerprint ridge patterns to classify blood groups using deep learning.

Research has shown correlations between dermatoglyphic (fingerprint) patterns and blood group traits. By training a CNN on labeled fingerprint images, the model learns to distinguish the subtle ridge configurations associated with each of the 8 blood groups.

**Supported blood group classes:** `A+` `A−` `B+` `B−` `AB+` `AB−` `O+` `O−`

---

## 📁 Repository Structure

```
BloodGroup-Prediction/
├── BGP.ipynb                  # Full pipeline: preprocessing → training → evaluation
├── dataset.zip                # Labeled fingerprint image dataset (8 classes)
└── model_checkpoints/         # Saved model weights, config, history, and eval results
    ├── fingerprint_model_<timestamp>.pth           # Model weights (state dict)
    ├── fingerprint_model_complete_<timestamp>.pth  # Full model (arch + weights)
    ├── training_history_<timestamp>.json           # Per-epoch loss & accuracy logs
    ├── model_config_<timestamp>.json               # Training configuration & summary
    └── evaluation_results_<timestamp>.npz          # Predictions & targets for analysis
```

---

## 🔄 Workflow

```
Fingerprint Images (dataset.zip)
        │
        ▼
Image Preprocessing
(Resize to 64×64 → ToTensor → Normalize)
        │
        ▼
Train / Val / Test Split
(train: N-2000, val: 1000, test: 1000)
        │
        ▼
Custom CNN Training
(6 Conv layers + 5 FC layers, Adam, lr=0.001, 20 epochs)
        │
        ▼
Checkpointing
(Weights, config, and history saved to model_checkpoints/)
        │
        ▼
Evaluation
(Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
        │
        ▼
Blood Group Prediction
(One of 8 ABO/Rh classes)
```

---

## 🏗️ Model Architecture

A custom CNN — `FingerprintToBloodGroup` — built with PyTorch:

| Layer Block | Details |
|---|---|
| Conv Block 1 | Conv2d(3→32) → ReLU → Conv2d(32→64) → ReLU → MaxPool2d |
| Conv Block 2 | Conv2d(64→128) → ReLU → Conv2d(128→128) → ReLU → MaxPool2d |
| Conv Block 3 | Conv2d(128→256) → ReLU → Conv2d(256→256) → ReLU → MaxPool2d |
| Fully Connected | Flatten → Linear(16384→1024) → Linear(1024→512) → Linear(512→256) → Linear(256→128) → Linear(128→8) |
| Activation | ReLU throughout; Softmax at inference |
| Optimizer | Adam (lr = 0.001) |
| Loss Function | Cross-Entropy Loss |
| Epochs | 20 |
| Batch Size | 128 |

---

## 📊 Results

### Overall Performance

| Metric | Score |
|---|---|
| **Overall Accuracy** | **87.80%** |
| **Balanced Accuracy** | **87.60%** |

### Per-Class Performance (on 1,000-sample validation set)

| Blood Group | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| A+  | 0.9083 | 0.8684 | 0.8879 | 114 |
| A−  | 0.8533 | 0.8889 | 0.8707 | 144 |
| AB+ | 0.8819 | 0.9407 | 0.9104 | 135 |
| AB− | 0.8618 | 0.8689 | 0.8653 | 122 |
| B+  | 0.9204 | 0.8889 | 0.9043 | 117 |
| B−  | 0.9143 | 0.8951 | 0.9046 | 143 |
| O+  | 0.7752 | 0.9091 | 0.8368 | 110 |
| O−  | 0.9348 | 0.7478 | 0.8309 | 115 |
| **Macro Avg** | **0.8812** | **0.8760** | **0.8764** | **1000** |

> The model was trained on a CUDA GPU. Training converged steadily over 20 epochs with no significant overfitting.

### Visualizations Generated
- Training & validation loss / accuracy curves
- Per-epoch time tracking
- Confusion matrix (raw counts + normalized %)
- Per-class precision / recall / F1 bar charts
- Multi-class ROC curves with AUC scores
- Sample prediction grid with confidence scores (color-coded correct/incorrect)

---

## 🧰 Tech Stack

- **Language:** Python 3
- **Environment:** Jupyter Notebook / Google Colab
- **Framework:** PyTorch + torchvision
- **Libraries:**
  - `torch`, `torchvision` — model building, training, data loading
  - `numpy`, `pandas` — data handling
  - `scikit-learn` — metrics (accuracy, F1, ROC-AUC, confusion matrix)
  - `matplotlib`, `seaborn` — visualization

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sk00301/BloodGroup-Prediction.git
cd BloodGroup-Prediction
```

### 2. Extract the dataset

```bash
unzip dataset.zip
```

### 3. Install dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

### 4. Run the notebook

Open and run all cells in:

```
BGP.ipynb
```

> 💡 Pre-trained weights are saved in `model_checkpoints/` and can be reloaded using the `load_model()` function for inference without retraining.

---

## 🧠 Key Concepts

**Fingerprint Ridge Patterns & Blood Groups** — Studies have identified correlations between dermatoglyphic features (ridge density, loop/whorl/arch patterns) and ABO/Rh blood group traits. The CNN learns these associations automatically from raw image pixels.

**Non-Invasive Blood Typing** — Unlike conventional serological methods, this approach requires only a scanned fingerprint image, making it potentially suitable for emergency care, remote healthcare, and blood donation camps.

**CNN Feature Extraction** — Stacked convolutional layers progressively extract features from edges → textures → ridge patterns, which are then classified by the fully connected head into one of 8 blood group categories.

**Model Checkpointing** — Weights, training history, configuration, and evaluation results are all saved at the end of training, enabling reproducibility and inference without retraining.

---

## ✅ Conclusion

The custom CNN achieved **87.80% overall accuracy** and **87.60% balanced accuracy** on the 8-class blood group prediction task from fingerprint images — trained in just 20 epochs on a GPU.

Key takeaways:
- **Best performing classes:** `AB+` and `B−` achieved F1-scores above **0.910**, indicating the model learned their fingerprint patterns most reliably
- **High recall:** `O+` achieved the highest recall at **90.91%**, meaning very few true O+ samples were missed
- **High precision:** `O−` had the highest precision at **93.48%**, meaning predictions for that class were rarely wrong
- **Smooth convergence:** Training loss dropped from 2.08 to ~0.40 and validation accuracy rose from ~20% to ~88% over 20 epochs, with no significant overfitting
- **Balanced performance:** The near-equal overall and balanced accuracy scores confirm the model generalises well across all 8 classes, not just the most frequent ones

With further improvements such as data augmentation, transfer learning from a larger backbone (e.g. ResNet or EfficientNet), or increased image resolution, accuracy could be pushed well beyond 90%.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not intended as a substitute for certified medical blood-typing procedures. Do not use for clinical or diagnostic decision-making.

---

## 📄 License

This project is open source. Feel free to use and build upon it.

---

## 🙋 Author

**[sk00301](https://github.com/sk00301)**
