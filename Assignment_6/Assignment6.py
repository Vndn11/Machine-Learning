import numpy as np
import pandas as pd
import sys

def activate_sigmoid(input_val):
    """Calculate the sigmoid activation."""
    return 1 / (1 + np.exp(-input_val))

def calculate_regularized_cost(features, target, params, reg_strength):
    """Compute cost with L2 regularization."""
    sample_size = len(target)
    predictions = features.dot(params)
    activated = activate_sigmoid(predictions)
    loss_positive = target * np.log(activated)
    loss_negative = (1 - target) * np.log(1 - activated)
    penalty = (reg_strength / (2 * sample_size)) * np.sum(np.square(params[1:]))
    total_cost = -(np.sum(loss_positive + loss_negative) / sample_size) + penalty
    return total_cost

def calculate_gradient_with_regularization(features, target, params, reg_strength):
    """Compute gradients with L2 regularization."""
    sample_size = len(target)
    predictions = features.dot(params)
    activated = activate_sigmoid(predictions)
    errors = activated - target
    base_gradient = np.dot(features.T, errors) / sample_size
    base_gradient[1:] += (reg_strength / sample_size) * params[1:]
    return base_gradient

def perform_gradient_descent(features, target, starting_params, alpha, num_iterations, reg_strength):
    """Apply gradient descent with regularization."""
    params = starting_params
    history_of_cost = []
    
    for _ in range(num_iterations):
        gradient = calculate_gradient_with_regularization(features, target, params, reg_strength)
        params -= alpha * gradient
        cost = calculate_regularized_cost(features, target, params, reg_strength)
        history_of_cost.append(cost)
        
    return params, history_of_cost

def make_predictions(features, params):
    """Predict binary outcomes using logistic regression."""
    prediction_scores = np.dot(features, params)
    predicted_classes = activate_sigmoid(prediction_scores)
    return np.where(predicted_classes >= 0.5, 1, 0)

# Path manipulation to access data
dataset = pd.read_csv(f'data12.csv')

# Preparing the dataset
features = dataset.drop(columns=['Unnamed: 0', 'y'])
target = dataset['y']
num_features = features.shape[1]

# Hyperparameters
alpha = 0.01  # Learning rate
num_iterations = 300
reg_strength = 0.1  # Regularization parameter

# Add intercept term
features_with_intercept = np.hstack([np.ones((features.shape[0], 1)), features])

# Initialize parameters
starting_params = np.zeros(features_with_intercept.shape[1])

# Optimization
final_params, cost_history = perform_gradient_descent(features_with_intercept, target, starting_params, alpha, num_iterations, reg_strength)
print("Intercept:", final_params[0])
print("Optimized parameters with regularization:\n", final_params[1:])
print("Final loss:", cost_history[-1])

# Prediction and accuracy calculation
predictions = make_predictions(features, final_params[1:])
accuracy_score = np.mean(target == predictions)
print("Model accuracy:", accuracy_score)
