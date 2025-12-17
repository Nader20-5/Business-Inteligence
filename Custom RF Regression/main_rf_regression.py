# -*- coding: utf-8 -*-
"""
Main script to run custom Random Forest Regression
Implements the same workflow as the notebook but with custom implementations
"""

import sys
import io
# Fix encoding for Windows console
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from custom_random_forest import (
    RandomForestRegressor,
    calculate_r2_score,
    calculate_rmse,
    train_test_split_custom
)


def custom_standard_scaler(X):
    """Custom implementation of StandardScaler"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def transform_with_scaler(X, mean, std):
    """Transform data using saved mean and std"""
    return (X - mean) / std


def custom_one_hot_encode(data, categorical_cols):
    """Custom one-hot encoding"""
    encoded_data = data.copy()
    
    for col in categorical_cols:
        unique_vals = data[col].unique()
        # Drop first category to avoid multicollinearity
        for i, val in enumerate(unique_vals[1:]):
            encoded_data[f'{col}_{val}'] = (data[col] == val).astype(int)
        encoded_data = encoded_data.drop(columns=[col])
    
    return encoded_data


def custom_impute_median(X):
    """Custom median imputation for missing values"""
    X_imputed = X.copy()
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        mask = ~np.isnan(col)
        if np.any(~mask):
            median_val = np.median(col[mask])
            X_imputed[~mask, col_idx] = median_val
    return X_imputed


def create_polynomial_features(X, degree=2, interaction_only=False):
    """Create polynomial and interaction features"""
    n_samples, n_features = X.shape
    
    if interaction_only:
        # Only create interaction terms (no squared terms)
        new_features = [X]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                new_features.append(interaction)
    else:
        # Create polynomial features up to specified degree
        new_features = [X]
        if degree >= 2:
            # Add squared terms for important features
            squared = X ** 2
            new_features.append(squared)
    
    return np.hstack(new_features)


def main():
    print("=" * 60)
    print("CUSTOM RANDOM FOREST REGRESSION - FROM SCRATCH")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/7] Loading data...")
    dataset = pd.read_csv('D:\My-Github\Business-Inteligence\Regression models\student-por.csv')
    print(f"[OK] Data Loaded. Original Shape: {dataset.shape}")
    
    # Drop Walc and Dalc
    dataset = dataset.drop(columns=['Walc', 'Dalc'])
    print("[OK] Dropped columns 'Walc' and 'Dalc'.")
    
    # 2. Clean Data
    print("\n[2/7] Cleaning data...")
    initial_rows = len(dataset)
    dataset = dataset[dataset['G3'] != 0]
    dropped_rows = initial_rows - len(dataset)
    print(f"[OK] Removed {dropped_rows} rows with '0' grades (Anomalies cleaned).")
    
    X = dataset.drop(columns=['G3'])
    y = dataset['G3'].values
    
    # 3. Identify column types
    print("\n[3/7] Preprocessing features...")
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # 4. Handle categorical data with custom one-hot encoding
    X_encoded = custom_one_hot_encode(X, categorical_cols)
    print("[OK] Categorical data encoded (Custom One-Hot Encoding).")
    
    # Convert to numpy array
    X_processed = X_encoded.values.astype(float)
    
    # 5. Handle missing values with custom median imputation
    X_processed = custom_impute_median(X_processed)
    print("[OK] Missing values imputed (Custom Median Imputation).")
    
    # 6. Train-test split using custom function
    print("\n[4/8] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split_custom(
        X_processed, y, test_size=0.2, random_state=42
    )
    print("[OK] Dataset split into Train and Test sets.")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # 7. Feature scaling using custom scaler
    print("\n[5/8] Scaling features...")
    X_train, train_mean, train_std = custom_standard_scaler(X_train)
    X_test = transform_with_scaler(X_test, train_mean, train_std)
    print("[OK] Feature Scaling applied (Custom StandardScaler).")
    
    # 8. Feature Engineering - Use standard features
    print("\n[6/8] Preparing features...")
    print("Using standard features (removing polynomial expansion to reduce noise)...")
    # Using the standard scaled features
    X_train_final = X_train
    X_test_final = X_test
    print(f"[OK] Feature count: {X_train.shape[1]}")
    
    # 9. Train custom Random Forest with optimized parameters
    print("\n[7/8] Training Enhanced Random Forest Model...")
    print("Using optimized hyperparameters for maximum accuracy...")
    print("This may take several minutes...")
    regressor = RandomForestRegressor(
        n_estimators=10,       # DEBUG: 10 trees
        max_depth=15,          # Deeper trees to capture complex patterns
        min_samples_split=3,   # Allow more granular splits
        min_samples_leaf=1,    # More detailed leaf nodes
        max_features=None,     # Use ALL features (standard for Regression)
        random_state=42
    )
    regressor.fit(X_train_final, y_train)
    print("[OK] Model Training Complete.")
    
    # 10. Make predictions and evaluate
    print("\n[8/8] Evaluating model...")
    y_pred = regressor.predict(X_test_final)
    
    # Calculate metrics using custom functions
    r2 = calculate_r2_score(y_test, y_pred)
    rmse = calculate_rmse(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"R-Squared Score: {r2:.4f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
    print("=" * 60)
    
    # 10. Visualize results
    print("\n[BONUS] Creating visualization...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
             color='red', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Grade', fontsize=12)
    plt.ylabel('Predicted Grade', fontsize=12)
    plt.title('Custom Random Forest: Actual vs Predicted Grades', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('custom_rf_results.png', dpi=300, bbox_inches='tight')
    print("[OK] Visualization saved as 'custom_rf_results.png'")
    plt.show()
    
    print("\n" + "=" * 60)
    print("CUSTOM IMPLEMENTATION COMPLETE!")
    print("=" * 60)
    print("\nAll components implemented from scratch:")
    print("Decision Tree Regressor")
    print("Random Forest Regressor")
    print("Train-Test Split")
    print("Standard Scaler")
    print("One-Hot Encoder")
    print("Median Imputer")
    print("R² Score Calculator")
    print("RMSE Calculator")


if __name__ == "__main__":
    main()
