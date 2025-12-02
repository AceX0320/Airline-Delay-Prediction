# Airline Delay Prediction System

## Overview
This project implements a Machine Learning system to predict flight delays using Logistic Regression. It compares two regularization techniques:
- **L2 Regularization (Ridge)**: Standard regularization that penalizes large coefficients.
- **L1 Regularization (Lasso)**: Regularization that encourages sparsity (feature selection).

The system is built for an academic project based on "lect_02_03_linear_models.pdf" and uses the `filtered_flight_data.csv` dataset.

## Project Structure
- `airline_delay_prediction.py`: Main Python script for data processing, training, and evaluation.
- `filtered_flight_data.csv`: Dataset file (must be present in the directory).
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.
- **Artifacts (Generated after running):**
    - `logistic_l2_model.pkl`: Trained L2 model.
    - `logistic_l1_model.pkl`: Trained L1 model.
    - `feature_scaler.pkl`: Scaler for data normalization.
    - `model_config.json`: Model configuration and feature names.
    - `eda_*.png`: Exploratory Data Analysis visualizations.
    - `model_roc_comparison.png`: ROC Curve comparison plot.

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

5.  **Run the prediction system**:
    ```bash
    python airline_delay_prediction.py
    ```

6.  **Run the GUI Simulation**:
    ```bash
    python delay_prediction_gui.py
    ```
    - **Dashboard View**: View all flight scenarios at a glance with the "Cool Blue" professional theme.
    - **Weather Simulation**: See how different weather conditions (Rain, Snow, Storm) impact delay probabilities.
    - **Interactive Analysis**: Hover over rows to highlight them. Click a flight to see detailed reasoning.
    - **Explainability**: The "Reasoning" panel explains *why* a flight is delayed, including weather impacts and factor descriptions.

7.  **View the results**:
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

## Model Details
- **Algorithm**: Logistic Regression (Sigmoid activation, Cross-Entropy Loss).
- **Optimization**: Stochastic Gradient Descent (solver='saga').
- **Regularization**:
    - L2: $\lambda ||w||^2_2$ (Prevents overfitting by shrinking weights).
    - L1: $\lambda ||w||_1$ (Performs feature selection by zeroing out weights).
