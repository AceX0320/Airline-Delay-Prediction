# Airline Delay Prediction System

## Overview
This project implements a Machine Learning system to predict flight delays using Logistic Regression. It compares two regularization techniques:
- **L2 Regularization (Ridge)**: Standard regularization that penalizes large coefficients.
- **L1 Regularization (Lasso)**: Regularization that encourages sparsity (feature selection).

The system is built for an academic project based on "lect_02_03_linear_models.pdf" and uses the `filtered_flight_data.csv` dataset.

## Project Structure
- `airline_delay_prediction.py`: Main Python script for data processing, training, and evaluation.
- `delay_prediction_gui.py`: Interactive GUI for flight simulation and analysis.
- `filtered_flight_data.csv`: Dataset file (must be present in the directory).
- `generate_data.py`: Script to create the synthetic dataset if missing.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.
- **Artifacts (Generated after training):**
    - `logistic_l2_model.pkl`: Trained L2 model.
    - `logistic_l1_model.pkl`: Trained L1 model.
    - `feature_scaler.pkl`: Scaler for data normalization.
    - `model_config.json`: Model configuration and feature names.
    - `eda_delay_distribution.png`: Exploratory Data Analysis visualization.
    - `eda_correlation_heatmap.png`: Feature correlation heatmap.
    - `model_roc_comparison.png`: ROC Curve comparison plot.
    - `model_confusion_matrices.png`: Confusion matrices for both models.
    - `feature_importance_l2.png`: Visual ranking of top features.

## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`

## Quick Start Guide

Follow this exact sequence of commands to set up and run the project:

1.  **Open your terminal** in the project directory.

2.  **(Optional) Create a virtual environment** to keep dependencies isolated:
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    - Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - Mac/Linux:
      ```bash
      source venv/bin/activate
      ```

4.  **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Generate Synthetic Data** (Required for first run):
    ```bash
    python generate_data.py
    ```

6.  **Run the prediction system** (Train Models):
    ```bash
    python airline_delay_prediction.py
    ```

7.  **Run the GUI Simulation**:
    ```bash
    python delay_prediction_gui.py
    ```
    - **Dashboard View**: View flight scenarios with the "Cool Blue" professional theme.
    - **Real-world Heuristics**: The system applies logic to simulate weather impacts missing from the raw dataset:
        - **Snow**: +30% Delay Risk
        - **Rain**: +15% Delay Risk
        - **Storm**: +45% Delay Risk
    - **Interactive Analysis**: Select a flight and click **"Predict"** to open the **Prediction Analysis Popup**.
    - **Explainability**: The popup breaks down the *why* behind a delay:
        - **Top Factors**: Ranked list of features contributing to the decision.
        - **Impact Score**: Shows exactly how much each factor (Weather, Airline, Time) pushes the probability up or down.

8.  **View the results**:
    - Check the terminal output for the model comparison report.
    - Open the generated images (`eda_delay_distribution.png`, `model_roc_comparison.png`, etc.) to see the visualizations.

## Output
The script will:
1.  Load and analyze the dataset.
2.  Perform Exploratory Data Analysis (EDA) and save plots.
3.  Preprocess data (cleaning, encoding, scaling).
4.  Train both L2 and L1 Logistic Regression models.
5.  Evaluate models and print a detailed comparison report (Accuracy, Precision, Recall, F1, ROC-AUC).
6.  Display the top influential features for each model.
7.  Save the trained models and artifacts for future use.

## Technical Highlights
This project demonstrates several advanced Applied AI concepts:

- **Regularization Strategy**:
    - **L2 (Ridge)** is used to handle multicollinearity (e.g., correlations between Distance and Air Time) and prevent overfitting by distributing weights.
    - **L1 (Lasso)** is implemented to perform automatic *feature selection*, zeroing out irrelevant features to simplify the model.

- **Class Imbalance Handling**:
    - Real-world flight data is heavily imbalanced (~80% on-time). The model uses `class_weight='balanced'` to adjust the loss function, ensuring the model prioritizes detecting potential delays rather than just maximizing trivial accuracy.

- **Explainable AI (XAI)**:
    - The GUI includes a custom reasoning engine that calculates the dot product of the individual flight's data vector and the model's coefficients ($w \cdot x$).
    - This allows for precise, per-flight explanations (e.g., "This flight is delayed specifically because of the Snow storm and the congestion at ORD," rather than just a generic probability).

## Model Details
- **Algorithm**: Logistic Regression (Sigmoid activation, Cross-Entropy Loss).
- **Optimization**: Stochastic Gradient Descent (solver='saga').
- **Regularization**:
    - L2: $\lambda ||w||^2_2$
    - L1: $\lambda ||w||_1$
