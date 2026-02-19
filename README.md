# ATP Tennis Data Analysis

This project analyzes ATP tennis match data to predict match outcomes.

## Project Structure

- `src/data/`: Contains raw and processed data.
- `src/notebooks/`: Jupyter notebooks for EDA and legacy data cleaning.
- `src/utils/`: Helper functions.
- `src/model/`: Trained models and result metrics.
- `Instrucciones/`: Project documentation and data descriptions.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost
    ```

2.  **Data Processing:**
    Open and run `src/notebooks/Limpieza.ipynb`.
    This notebook reads from `src/data/raw`, cleans the data, and saves the result to `src/data/processed/dfLimpio.csv`.

3.  **Model Training:**
    To train the models (Random Forest, AdaBoost, Logistic Regression, Bagging, XGBoost):
    ```bash
    python src/train.py
    ```
    This script loads the processed data, splits it into Train (80%) and Test (20%) sets, trains the models, saves them to `src/model/`, and outputs the accuracy metrics.

## Results

Current model performance (Accuracy on Test Set):

| Model               | Accuracy |
|---------------------|----------|
| AdaBoost            | 64.5%    |
| Random Forest       | 64.2%    |
| Logistic Regression | 63.7%    |
| Bagging             | 62.8%    |
| XGBoost             | 61.3%    |

*Note: The baseline accuracy for random guessing is 50%.*

## Key Improvements

- **Data Leakage Fix**: The original data structure used `winner_` and `loser_` columns, which are not available before a match. The data has been restructured to `player_1` and `player_2` with a binary target.
- **Validation**: Added a proper Train/Test split to evaluate models on unseen data, preventing overfitting (previous scores were falsely 1.0).
- **Refactoring**: Code modularized into `process_data_v2.py` and `funciones.py`, with hardcoded paths removed for portability.
