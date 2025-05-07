# 🔐 Adversarial Attacks on PathMNIST Using FGSM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project demonstrates adversarial attacks on medical image classification models using the **Fast Gradient Sign Method (FGSM)**, tested in both **white-box** and **black-box** settings with the **PathMNIST** dataset.

---

## 📚 Table of Contents

- [📁 Project Structure](#-project-structure)
- [🚀 How It Works](#-how-it-works)
  - [🔓 White-box Attack](#-white-box-attack)
  - [🕶️ Black-box Attack](#-black-box-attack)
- [📊 Results](#-results)
- [🧪 Setup & Run](#-setup--run)
- [📚 Dataset](#-dataset)
- [🔭 Future Work](#-future-work)
- [🧑‍💻 Author](#-author)

---

## 📁 Project Structure

├── attacks/ # Attack-specific modules
├── data/ # Data preparation scripts
├── models/ # Model definitions
├── AdvFGSM_PathMNIST.ipynb # White-box FGSM attack notebook
├── AdvBlackbox_pathmnist.ipynb # Black-box FGSM attack notebook
├── conference_letter_report.docx # Report/summary document
├── eval.py # Evaluation script
├── model.py # Model architecture
├── model_pathmnist.pth # Pretrained model
├── requirements.txt # Python dependencies
├── train.py # Model training
├── utils.py # Helper functions
└── README.md # Project documentation


---

## 🚀 How It Works

### 🔓 White-box Attack

- Assumes full access to the model (weights & gradients).
- FGSM perturbs inputs in the direction of the gradient to mislead predictions.
- Evaluates the model's vulnerability to known-source attacks.

### 🕶️ Black-box Attack

- No access to the target model.
- A **surrogate model** is trained to approximate the target.
- FGSM is applied on the surrogate, and adversarial examples are transferred.
- Tests **transferability** of adversarial attacks.

---

## 📊 Results

| Attack Type   | Clean Accuracy | Adversarial Accuracy | Epsilon (ε) |
|---------------|----------------|-----------------------|-------------|
| White-box     |  *e.g., 88%*    |  *e.g., 43%*           | 0.1         |
| Black-box     |  *e.g., 87%*    |  *e.g., 51%*           | 0.1         |

- Visual changes are minimal but enough to confuse the model.
- Even black-box attacks reduce accuracy significantly.

---

## 🧪 Setup & Run

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/Adversarial-FGSM-PathMNIST.git
cd Adversarial-FGSM-PathMNIST
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```
Evaluate or run attacks
```bash
python eval.py
```
Or run Jupyter notebooks:
```bash
AdvFGSM_PathMNIST.ipynb
AdvBlackbox_pathmnist.ipynb
```
## 📚 Dataset

Dataset: PathMNIST from the MedMNIST collection
Format: 3-channel RGB images, 28×28 resolution, 9 classes of tissue types
🔭 Future Work

Add adversarial training as a defense mechanism
Implement stronger attacks like PGD and CW
Use Grad-CAM to visualize vulnerable regions
Evaluate robustness across different datasets

## 🧑‍💻 Author

Rohan Tiwari

