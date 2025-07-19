# ⚾ Baseball Win Prediction Model

This repository contains a Python script that implements a machine learning pipeline to predict baseball team wins (`W`) based on various team statistics. The solution employs extensive feature engineering, data scaling, and an advanced ensemble modeling technique (blending/stacking) to achieve robust predictions.

---

## 📌 Project Overview

The core objective of this project is to **accurately predict the number of wins for baseball teams**. The script processes historical team data, generates new insightful features, and trains a sophisticated ensemble model to make predictions. 

🏅 **Final Private Leaderboard MAE: 2.53786** (Kaggle Competition)

The final output is a submission file compatible with typical machine learning competitions.

---

## ✨ Features

### 🔧 Comprehensive Feature Engineering

- Calculates **Batting Average (BA)** and **Walks plus Hits per Inning Pitched (WHIP)**.
- Derives **per-game statistics** for various offensive and defensive metrics (Runs, Home Runs, Walks, Strikeouts, etc.).
- Introduces composite features like:
  - **"OPS-like"** (BA + BB_per_game)
  - **K/BB ratio**
  - **ERA/WHIP**

### 🧼 Robust Data Preprocessing

- Handles missing values by filling them with zeros.
- Applies `StandardScaler` to normalize features, ensuring models perform optimally.

### 🤖 Advanced Ensemble Modeling (Blending/Stacking)

- **Base Models**:
  - `RidgeCV`
  - `RandomForestRegressor`
- Implements a `StackingRegressor` with `RidgeCV` as the final estimator.
- Uses **5-fold `GroupKFold` cross-validation**, grouping by `yearID`, to generate out-of-fold (OOF) predictions for the meta-learner.  
  This helps prevent data leakage and provides a more reliable estimate of model performance.
- A `LinearRegression` model acts as the **meta-learner**, blending the predictions from the base models.

### 📁 Submission File Generation

- Generates a `submission_meta_blend.csv` file with predicted `'W'` values for the test dataset, formatted for competition submission.

---

## 🛠️ Installation

To run this script, you'll need to create the `moneyball` environment:

```bash
conda env create -f environment.yml
```

---

## 🚀 Usage

### ▶️ Run the Script

Navigate to the directory containing the Python script and execute:

```bash
cd src
python pipeline.py
```

This will perform:
- Feature engineering
- Model training
- Visualization of model performance on the last validation fold, including:
  - **Actual vs Predicted Wins** scatter plot
  - **Residual Plot** (Prediction Errors vs Predicted Wins)
- Generation of `submission_meta_blend.csv` in the `data` directory.

---

## 📊 Model Details

### 🧱 Model Architecture: Blending/Stacking Ensemble

#### ✅ Base Models:
- **RidgeCV**:
  - Ridge Regression with built-in cross-validation to select the optimal alpha.
  - Robust to multicollinearity.
- **RandomForestRegressor**:
  - Ensemble tree-based model.
  - Captures non-linear relationships and interactions.

#### 🧠 Meta-Learner:
- **LinearRegression**:
  - Trained on the **out-of-fold predictions** of the base models.
  - Learns the optimal weights to combine the predictions from Ridge and Random Forest, leveraging their strengths.

#### 📅 Cross-Validation Strategy:
- Uses `GroupKFold` split by `yearID`.
- Ensures no data leakage across years in the train-validation splits.
- Provides realistic model performance assessment on unseen years.

---
