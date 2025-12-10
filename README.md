# Airline Delay Prediction System

## Overview
This project implements a Machine Learning system to predict flight delays using Logistic Regression. It employs a **Hybrid Intelligence** approach, combining a rigorously trained statistical model with a rule-based expert system to account for real-time factors (like weather) missing from historical datasets.

## Project Structure
- `airline_delay_prediction.py`: Main Python script for data processing, training, and evaluation.
- `delay_prediction_gui.py`: Interactive GUI for flight simulation and analysis.
- `prediction_utils.py`: **[NEW]** Shared utility module containing the core prediction and explanation logic.
- `filtered_flight_data.csv`: Dataset file (must be present in the directory).
- `generate_data.py`: Script to create the synthetic dataset if missing.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.
- **Artifacts (Generated after training):**
    - `logistic_l2_model.pkl`: Trained L2 model (Backup).
    - `logistic_l1_model.pkl`: Trained L1 model (Active Core).
    - `feature_scaler.pkl`: Scaler for data normalization.
    - `model_config.json`: Model configuration and feature names.
    - `eda_delay_distribution.png`: Exploratory Data Analysis visualization.
    - `eda_correlation_heatmap.png`: Feature correlation heatmap.
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
    - **Hybrid Logic**: The system applies heuristic logic to simulate weather impacts:
        - **Snow**: +30% Delay Risk
        - **Rain**: +15% Delay Risk
    - **Interactive Analysis**: Select a flight and click **"Predict"** to open the **Prediction Analysis Popup**.
    - **Explainability**: The popup breaks down the *why* behind a delay using **Marginal Probability Contribution**:
        - **Unified Metrics**: Statistical factors (like airline history) and heuristic factors (like snow) are both converted to **Probability Impact %**, allowing for a direct apples-to-apples comparison.

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
    - **L1 (Lasso)** is used as the primary model to perform automatic *feature selection*, zeroing out irrelevant features to simplify the model.

- **Hybrid AI Architecture**:
    - Combines a **Data-Driven Model** (Logistic Regression) for historical patterns with a **Rule-Based System** (Heuristics) for dynamic external factors like weather.

- **Explainable AI (XAI)**:
    - **Marginal Probability Contribution**: instead of showing raw coefficients (Log-Odds), the system calculates the approximate change in probability ($\beta_i x_i \times p(1-p)$) caused by each feature. This makes the explanation intuitive for non-technical users (e.g., "Saturday travel increased risk by 5.2%").

## Model Details
- **Algorithm**: Logistic Regression (Sigmoid activation, Cross-Entropy Loss).
- **Optimization**: Stochastic Gradient Descent (solver='saga').
- **Regularization**: L1: $\lambda ||w||_1$
