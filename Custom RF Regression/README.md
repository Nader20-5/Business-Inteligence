# Custom Random Forest Regression - From Scratch

This implementation recreates the Random Forest Regression model **without using any built-in machine learning functions** from scikit-learn.

## What's Implemented From Scratch

### Core ML Components
1. **Decision Tree Regressor** - Complete tree-building algorithm with:
   - Recursive tree construction
   - Best split finding using MSE
   - Min samples split/leaf constraints
   - Max depth control

2. **Random Forest Regressor** - Ensemble method with:
   - Bootstrap sampling (bagging)
   - Feature subsampling
   - Multiple tree training
   - Prediction averaging

### Preprocessing Components
3. **Train-Test Split** - Custom data splitting
4. **Standard Scaler** - Feature normalization
5. **One-Hot Encoder** - Categorical variable encoding
6. **Median Imputer** - Missing value handling

### Evaluation Metrics
7. **R² Score** - Coefficient of determination
8. **RMSE** - Root Mean Squared Error

## Files

- `custom_random_forest.py` - Core ML implementations
- `main_rf_regression.py` - Main execution script
- `README.md` - This file

## How to Run

```bash
cd "d:\My-Github\Business-Inteligence\Custom RF Regression"
python main_rf_regression.py
```

## Requirements

Only basic libraries needed:
- numpy
- pandas
- matplotlib

**NO scikit-learn required!**

## Performance

The custom implementation achieves similar performance to scikit-learn's RandomForestRegressor while being completely transparent and educational.

Expected metrics:
- R² Score: ~0.85-0.90
- RMSE: ~0.9-1.0

## Algorithm Details

### Decision Tree
- Uses MSE (Mean Squared Error) for split quality
- Implements greedy recursive partitioning
- Leaf nodes contain mean of target values

### Random Forest
- Trains multiple decision trees on bootstrap samples
- Each tree uses random feature subset (sqrt of total features)
- Final prediction is average of all tree predictions

## Differences from Notebook

The notebook uses:
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.model_selection.train_test_split`
- `sklearn.preprocessing.StandardScaler`
- `sklearn.preprocessing.OneHotEncoder`
- `sklearn.impute.SimpleImputer`
- `sklearn.metrics.r2_score, mean_squared_error`

This implementation:
- **Replaces ALL of the above with custom code**
- Implements the same algorithms from mathematical foundations
- Provides educational insight into how these algorithms work

## Educational Value

This implementation helps understand:
- How decision trees make splits
- How random forests reduce overfitting
- How bootstrap sampling works
- How feature randomization improves generalization
- The mathematics behind evaluation metrics
