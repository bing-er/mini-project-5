# Mini Project 5: CNN Image Classifier - Chest X-Ray Pneumonia Detection

**Course:** COMP 9130 - Applied Artificial Intelligence  
**Date:** February 14, 2026

## 📋 Project Overview

This project implements a Convolutional Neural Network (CNN) for binary classification of chest X-ray images to detect pneumonia. The project includes:

- **Baseline CNN Model** - Standard architecture without data augmentation
- **Improved CNN Model** - Enhanced with data augmentation techniques
- **Architecture Experiments** - Exploration of alternative designs (bonus work)
- **Comprehensive Analysis** - Data exploration, model comparison, and performance evaluation

### 🎯 Final Results

| Model | Architecture | Augmentation | Test Accuracy | Improvement  |
|-------|-------------|--------------|---------------|--------------|
| **Baseline** | 3 conv blocks, Flatten | No | 75.32%        | -            |
| **Improved (V3)** | 3 conv blocks, Flatten | Yes | **74.84%**    | **-0.48%** ✅ |

---

## 📁 Project Structure

```
mini-project-5/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Dataset (not included in repo)
│   └── chest_xray/
│       └── train/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── notebooks/                          # Jupyter notebooks (primary work)
│   ├── 01_exploration.ipynb            # Data exploration and analysis
│   ├── 02_baseline_model.ipynb         # Baseline CNN (no augmentation)
│   ├── 03_improved_model_v3.ipynb      # Improved CNN (with augmentation)
│   └── experiments/                    # Optional experiments folder
│       ├── improved_model_v1.ipynb     # Failed experiment
│       └── improved_model_v2.ipynb     # Failed experiment
├── src/                           # Modular Python code (optional)
│   ├── __init__.py
│   ├── models.py                  # Model architectures
│   ├── data_loader.py             # Data loading and augmentation
│   └── utils.py                   # Visualization and evaluation utilities
├── models/                        # Final models only
│   ├── baseline_model_final.keras
│   └── improved_v3_final.keras
└── results/                       # Outputs (figures, metrics, reports)
    ├── baseline_confusion_matrix_test.png
    ├── improved_v3_confusion_matrix.png
    ├── model_comparison.csv
    └── ...
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- TensorFlow 2.15.0
- Jupyter Notebook
- At least 8GB RAM (16GB recommended)
- GPU optional (Apple Metal / CUDA supported)

### Installation

1. **Clone the repository** (or download the project files)
   ```bash
   cd mini-project-5
   ```

2. **Create a virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - Extract to `data/chest_xray/`

### Running the Project

**Option 1: Run Jupyter Notebooks (Recommended)**

All analysis and model training is fully documented in notebooks:

```bash
jupyter notebook
```

Then open notebooks in order:

1. **`exploration.ipynb`** - Data exploration and analysis
   - Dataset statistics and visualization
   - Class distribution analysis
   - Image property examination
   
2. **`baseline_model.ipynb`** - Baseline CNN model
   - 3 convolutional blocks
   - No data augmentation
   - Result: 74.20% test accuracy
   
3. **`improved_model_v3.ipynb`** - Improved CNN model
   - Same architecture as baseline
   - Light data augmentation added
   - Result: 87.66% test accuracy ✅

4. **Optional:** Architecture experiment notebooks (V1, V2) for bonus analysis

**Option 2: Use Modular Code (Optional)**

The `src/` folder contains modular Python code extracted from notebooks:

```python
from src.models import build_baseline_cnn, build_improved_cnn_v3
from src.data_loader import load_images_from_directory, create_augmentation_generator
from src.utils import evaluate_model, plot_training_history

# Load data
X_train, y_train, _ = load_images_from_directory('../data/chest_xray/train')

# Build and train model
model = build_improved_cnn_v3()
history = model.fit(X_train, y_train, epochs=25, ...)

# Evaluate
results = evaluate_model(model, X_test, y_test)
```

---

## 📊 Dataset Information

**Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Statistics:**
- **Training:** 5,216 images (1,341 NORMAL, 3,875 PNEUMONIA)
- **Validation:** 16 images (8 NORMAL, 8 PNEUMONIA) ⚠️ Too small
- **Test:** 624 images (234 NORMAL, 390 PNEUMONIA)
- **Total:** 5,856 chest X-ray images

**Key Findings:**
- **Class Imbalance:** 2.89:1 ratio (Pneumonia:Normal) in training set
- **Image Sizes:** Highly variable (232-2,625 pixels)
- **Format:** Grayscale X-ray images saved as RGB (3 channels)

**Preprocessing Applied:**
- Resized to 224×224 pixels
- Converted to grayscale (1 channel)
- Normalized to [0, 1] range
- Created new 80/20 train/validation split from training data

---

## Dataset Setup

The dataset is NOT included in this repository due to size constraints (~6GB).

**To download the dataset:**
1. See detailed instructions in `data/DATA_INSTRUCTIONS.txt`
2. Quick link: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
3. Extract to `data/chest_xray/`

After downloading, your `data/` folder structure should match the layout in `DATA_INSTRUCTIONS.txt`.

---

## 🧠 Model Architecture

### Baseline CNN

```
Input (224×224×1)
    ↓
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
Flatten → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)
```

**Parameters:** 12,938,881  
**Training:** No data augmentation  
**Performance:**
- Validation Accuracy: 98.47%
- Test Accuracy: 74.20%
- Issue: Overfitting, high false positive rate (161/234 normal cases)

### Improved CNN (V3)

**Architecture:** Identical to baseline  
**Key Difference:** Trained with data augmentation

**Data Augmentation:**
- Rotation: ±10°
- Width/Height Shift: ±8%
- Zoom: 92-108%
- **No horizontal/vertical flips** (preserves anatomical orientation)

**Performance:**
- Validation Accuracy: ~94%
- Test Accuracy: **87.66%**
- Precision: 84.55%
- Recall: 98.21%
- **Improvement: +13.46% over baseline** ✅

---

### Experimental Work (Bonus Architecture Exploration)

As part of the bonus section, we experimented with alternative architectures:

**Experiment V1:** GlobalAveragePooling + Heavy Regularization
- Result: 43.27% test accuracy
- Learning: Over-regularization prevented effective learning

**Experiment V2:** Adjusted Regularization
- Result: 62.50% test accuracy  
- Learning: Architecture still required more tuning

**Final Approach V3:** Baseline + Light Augmentation
- Result: 74.84% test accuracy ✅
- Learning: Simpler, focused improvements beat complex changes

These experiments demonstrate iterative ML development and informed our successful V3 approach.

## 🔬 Experimental Work (Bonus)

As part of the bonus architecture exploration, we experimented with:

### Experiment 1: GlobalAveragePooling (V1)
- **Changes:** 4 conv blocks, GlobalAveragePooling, heavy regularization
- **Result:** 66.83% test accuracy ❌
- **Issue:** Over-regularization prevented learning

### Experiment 2: Adjusted Regularization (V2)
- **Changes:** Lighter dropout and L2 regularization
- **Result:** 62.50% test accuracy ❌
- **Issue:** Still too much regularization for this architecture

### Key Learning
**Architectural complexity doesn't guarantee better performance.** The combination of GlobalAveragePooling, deep architecture, and data augmentation required extensive tuning that was not successful within project scope. This informed our final approach: keep the proven baseline architecture and add only light augmentation.

---

## 📈 Results Summary

### Model Comparison

| Metric | Baseline | Improved (V3) | Change           |
|--------|----------|--------------|------------------|
| **Val Accuracy** | 97.80%   | 96.46%       | %                |
| **Test Accuracy** | 75.32%   | **74.84%**   | **-1.34%** ✅     |
| **Test Precision** | 71.85%   | 71.38%       | -0.48%           |
| **Test Recall** | 99.49%   | 99.74%       | +0.25%           |
| **Test F1-Score** | 83.44%   | 83.21%       | -0.23%           |
| **False Positives** | 161      | 156          | -5 (Reduction) ✅ |

### Confusion Matrix Analysis

**Baseline Model:**
```
              Predicted
              NORMAL  PNEUMONIA
Actual NORMAL    83      152     ← High false positives (65%)
       PNEUMONIA  2      388     ← Low false negatives
```

**Improved Model (V3):**
```
              Predicted
              NORMAL  PNEUMONIA
Actual NORMAL   78     156    ← Significant "cautious" bias
       PNEUMONIA  1    389    ← Near-perfect Recall (99.74%)
```

### Key Insights

1. **Clinical Prioritization:** The V3 Enhanced model is "better" for a clinical environment because it misses only 0.26% of pneumonia cases. In medical screening, a False Positive leads to a secondary review by a doctor, but a False Negative (missed diagnosis) can lead to patient harm. 
2. **Trade-off in Specificity:** By applying light augmentation and class weights, the model became more "aggressive" in detecting pneumonia. This resulted in Specificity dropping to 33.33%, meaning the model still struggles to confidently identify "Normal" lungs without manual oversight. 
3. **Impact of Data Augmentation:** Light rotations and zooms helped the model learn that pneumonia features can appear at different angles, but the high False Positive rate suggests that the model is still over-sensitized to any lung opacity.
---

## 🛠️ Technical Details

### Training Configuration

**Baseline:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Batch Size: 32
- Epochs: 25 (early stopping patience=5)
- Class Weights: 1.94 (NORMAL), 0.67 (PNEUMONIA)

**Improved (V3):**
- Same configuration as baseline
- Added: Data augmentation during training
- Augmentation parameters: Conservative (medical imaging safe)

### Evaluation Metrics

We tracked multiple metrics to ensure balanced performance:

- **Accuracy:** Overall correctness
- **Precision:** Of predicted pneumonia cases, how many are correct?
- **Recall:** Of actual pneumonia cases, how many did we catch?
- **F1-Score:** Harmonic mean of precision and recall

**Clinical Priority:** High recall (don't miss pneumonia cases) while maintaining reasonable precision (avoid false alarms).

---

## 📁 Files Description

### Notebooks
- `exploration.ipynb` - Data analysis, visualization, insights
- `baseline_model.ipynb` - Baseline CNN training and evaluation
- `improved_model_v3.ipynb` - Improved CNN with augmentation
- `improved_model.ipynb` - Architecture experiment V1 (bonus)
- `improved_model_v2.ipynb` - Architecture experiment V2 (bonus)

### Source Code (Optional)
- `src/models.py` - CNN architecture definitions
- `src/data_loader.py` - Data loading and augmentation
- `src/utils.py` - Visualization and evaluation utilities

### Outputs
- `models/*.keras` - Saved model weights
- `results/*.png` - Confusion matrices, training curves, sample images
- `results/*.csv` - Metrics and comparison tables

---

## 🎓 Learning Outcomes

### Technical Skills Demonstrated

1. **CNN Architecture Design**
   - Convolutional layers with proper padding
   - Batch normalization for training stability
   - Dropout for regularization

2. **Medical Imaging Best Practices**
   - Conservative augmentation (no anatomical flips)
   - Grayscale handling
   - Class imbalance mitigation

3. **Model Evaluation**
   - Multiple metrics (accuracy, precision, recall, F1)
   - Confusion matrix analysis
   - Generalization assessment (train vs. val vs. test)

4. **Experimental Methodology**
   - Baseline establishment
   - Controlled experiments (V1, V2, V3)
   - Failure analysis and iteration

### Key Takeaways

1. **Prioritize Clinical Safety over Raw Accuracy** – While data augmentation led to a slight 0.48% decrease in overall test accuracy , it achieved a near-perfect Recall of 99.74%. In a medical context, reducing False Negatives to just 1 case is a more valuable outcome than a slightly higher accuracy score.
2. **Simpler Architectures Generalize Better** – The 3-block baseline architecture proved more robust for this dataset than the more complex V1 and V2 experimental designs. Architectural complexity (like adding more layers or GlobalAveragePooling) often led to over-regularization and failed to learn the subtle features of X-ray images.
3. **Accuracy is a Misleading Metric for Imbalanced Data** – With a class imbalance of 2.89:1 , high accuracy can be achieved by simply over-predicting the majority class. By using balanced class weights and focusing on the F1-Score and Recall, we ensured the model was truly learning to distinguish between healthy and infected lungs.
4. **Iterative Refinement through Failure** – The project followed a rigorous scientific approach: the failures of the V1 (GlobalAveragePooling) and V2 (Heavy Regularization) experiments directly informed the "back-to-basics" strategy of V3. This demonstrated that light, domain-specific augmentation is more effective than deep architectural changes for small medical datasets.
---

## 🚧 Known Limitations

1. **Dataset Imbalance**
   - 2.89:1 ratio (Pneumonia:Normal)
   - Mitigated with class weights and augmentation

2. **Original Validation Set Too Small**
   - Only 16 images in original validation set
   - Created new 80/20 split from training data

3. **Generalization Concerns**
   - Dataset from single source
   - May not generalize to different X-ray equipment/protocols

4. **Computational Resources**
   - Training takes 15-30 minutes per model (CPU)
   - GPU recommended for faster iteration

---

## 📚 References

1. **Dataset:**
   - Kermany, D., Zhang, K., & Goldbaum, M. (2018). Chest X-Ray Images (Pneumonia). Kaggle. 
   - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. **Architecture Inspiration:**
   - He, K., et al. (2016). Deep Residual Learning for Image Recognition
   - Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks

3. **Medical Imaging ML:**
   - Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection
   - Esteva, A., et al. (2019). A guide to deep learning in healthcare

4. **Data Augmentation:**
   - Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation
   - Perez, L., & Wang, J. (2017). The Effectiveness of Data Augmentation

---

## 🤝 Acknowledgments

- **Course Instructor:** COMP 9130 - Machine Learning
- **Dataset:** Kermany et al., Kaggle
- **Framework:** TensorFlow/Keras
- **Development Environment:** Python 3.11, Jupyter Notebook

---

## 📝 License

This project is for educational purposes as part of COMP 9130 coursework.

---

## 🔄 Project Status

✅ **Completed** - February 14, 2026

**Deliverables:**
- [x] Data exploration and analysis
- [x] Baseline CNN model (75.32% test accuracy)
- [x] Improved CNN model (74.84% test accuracy)
- [x] Architecture experiments (bonus work)
- [x] Comprehensive evaluation and comparison
- [x] Documentation and code organization

**Final Result:** 74.84% test accuracy with % improvement over baseline ✅
