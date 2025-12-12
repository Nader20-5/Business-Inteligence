"""
Custom Random Forest Regression Implementation
Built from scratch without using sklearn's RandomForestRegressor
"""

import numpy as np
from collections import Counter


class DecisionTreeRegressor:
    """Custom Decision Tree for Regression"""
    
    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def fit(self, X, y):
        """Build the decision tree"""
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        """Recursively build the tree"""
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'value': np.mean(y)}
        
        # Find best split
        best_split = self._find_best_split(X, y, n_features)
        
        if best_split is None:
            return {'value': np.mean(y)}
        
        # Split the data
        left_idx = X[:, best_split['feature']] <= best_split['threshold']
        right_idx = ~left_idx
        
        # Check minimum samples per leaf
        if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
            return {'value': np.mean(y)}
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }
    
    def _find_best_split(self, X, y, n_features):
        """Find the best feature and threshold to split on - Optimized O(N)"""
        best_mse = float('inf')
        best_split = None
        n_samples = len(y)
        
        # Pre-calculate sums for the parent node (constant across features)
        sum_y = np.sum(y)
        sum_y2 = np.sum(y ** 2)
        
        # Try every feature
        for feature_idx in range(n_features):
            # Sort samples by feature value
            # This allows efficient linear scan for best split
            sorted_indices = np.argsort(X[:, feature_idx])
            X_sorted = X[sorted_indices, feature_idx]
            y_sorted = y[sorted_indices]
            
            # Linear scan variables
            left_n = 0
            left_sum = 0.0
            left_sum2 = 0.0
            
            right_n = n_samples
            right_sum = sum_y
            right_sum2 = sum_y2
            
            # Iterate through possible split points
            # We check split between i and i+1
            for i in range(n_samples - 1):
                y_val = y_sorted[i]
                
                # Update sums
                left_n += 1
                left_sum += y_val
                left_sum2 += y_val ** 2
                
                right_n -= 1
                right_sum -= y_val
                right_sum2 -= y_val ** 2
                
                # Skip if less than min_samples_leaf
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                
                # Skip if feature values are identical (no real split)
                if X_sorted[i] == X_sorted[i+1]:
                    continue
                    
                # Calculate MSE of split
                # Var = E[X^2] - (E[X])^2
                # MSE = Var (since we mean predict)
                # Weighted MSE = (n_left * Var_left + n_right * Var_right) / n_total
                
                left_var = (left_sum2 / left_n) - (left_sum / left_n) ** 2
                right_var = (right_sum2 / right_n) - (right_sum / right_n) ** 2
                
                # Handle precision issues
                if left_var < 0: left_var = 0
                if right_var < 0: right_var = 0
                
                mse = (left_n * left_var + right_n * right_var) / n_samples
                
                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature': feature_idx,
                        'threshold': (X_sorted[i] + X_sorted[i+1]) / 2 # Midpoint split
                    }
                    
        return best_split
    
    def _calculate_mse(self, left_y, right_y):
        """Calculate Mean Squared Error for a split"""
        left_mse = np.var(left_y) * len(left_y)
        right_mse = np.var(right_y) * len(right_y)
        total_mse = (left_mse + right_mse) / (len(left_y) + len(right_y))
        return total_mse
    
    def predict(self, X):
        """Predict values for X"""
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction"""
        if 'value' in node:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])


class RandomForestRegressor:
    """Custom Random Forest Regressor - Enhanced for Higher Accuracy"""
    
    def __init__(self, n_estimators=500, max_depth=15, min_samples_split=3, 
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        """Train the random forest"""
        self.trees = []
        n_samples, n_features = X.shape
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_idx]
            y_bootstrap = y[bootstrap_idx]
            
            # Feature subsampling
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            else:
                max_features = n_features
            
            feature_idx = np.random.choice(n_features, max_features, replace=False)
            X_subset = X_bootstrap[:, feature_idx]
            
            # Train tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, y_bootstrap)
            
            self.trees.append({
                'tree': tree,
                'features': feature_idx
            })
            
            if (i + 1) % 50 == 0:
                print(f"Trained {i + 1}/{self.n_estimators} trees")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            features = tree_info['features']
            X_subset = X[:, features]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        # Average predictions from all trees
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)


def calculate_r2_score(y_true, y_pred):
    """Calculate R² score manually"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_rmse(y_true, y_pred):
    """Calculate RMSE manually"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """Custom train-test split"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return X_train, X_test, y_train, y_test
