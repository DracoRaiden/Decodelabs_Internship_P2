# Data Classification Pipeline (K-Nearest Neighbors)

**DecodeLabs | Artificial Intelligence - Project 2**

## Overview
This project transitions from rule-based heuristics to **Supervised Learning**. Instead of hardcoding explicit `if-else` logic, this pipeline establishes a digital foundation that empowers the machine to recognize patterns and derive its own decision boundaries using the K-Nearest Neighbors (KNN) algorithm.

## Architecture (IPO Framework)
The model is built on a robust Input-Process-Output structure:
1. **Input (Raw Material):** Ingests the benchmark Iris Dataset, containing 150 samples distributed across 3 classes and 4 dimensional features (Sepal/Petal dimensions).
2. **Process (The Logic Skeleton):** * **Train/Test Split:** Randomizes and divides data into an 80% training set and a 20% validation set to prevent overfitting.
   * **Feature Scaling:** Employs a `StandardScaler` to normalize raw data into a balanced state (Mean = 0, Variance = 1).
   * **Algorithm:** Instantiates and fits a KNN classifier using an optimal neighbor count (`K=5`).
3. **Output (Validation):** Evaluates the model beyond the "accuracy mirage" of imbalanced data by generating a complete Confusion Matrix and calculating the F1 Score (the harmonic mean of Precision and Recall).

## Features
* **Algorithmic Logic:** Teaches a machine to categorize new information based on historical data.
* **Proximity Principle:** Classifies data points based on the majority vote of their nearest spatial neighbors.
* **Deterministic Guardrails:** Safely scales inputs before prediction to ensure dimensional equality.

## Usage
Ensure you have `scikit-learn` and `pandas` installed in your environment. Run the script via your Python terminal:

```bash
python main.py
