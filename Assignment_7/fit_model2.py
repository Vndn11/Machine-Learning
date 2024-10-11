import argparse
import csv
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def stringify_model_coefficients(coefficients, cosine_terms, sine_terms):
    """
    Formats the model coefficients into a readable string.
    """
    initial_coefficient = coefficients[0] if isinstance(coefficients[0], np.float64) else coefficients[0][0]
    intercept_term = f"{initial_coefficient:+.5f} "
    cosine_series = [f"{coefficients[i]:+.5f} cos(2π * {i})" for i in range(1, cosine_terms + 1)]
    sine_series = [f"{coefficients[i+cosine_terms]:+.5f} sin(2π * {i})" for i in range(1, sine_terms + 1)]
    return intercept_term + " ".join(cosine_series + sine_series)

def estimate_harmonic_model_coefficients(x, y, cosine_terms, sine_terms):
    """
    Fits a harmonic model to the provided data, returning the model coefficients and RMSE.
    """
    x = x.reshape(-1, 1)
    y = y.flatten()
    matrix = np.concatenate([
        np.ones_like(x),
        np.array([np.cos(2 * np.pi * i * x) for i in range(1, cosine_terms + 1)]).transpose(1, 0, 2).reshape(len(x), -1),
        np.array([np.sin(2 * np.pi * i * x) for i in range(1, sine_terms + 1)]).transpose(1, 0, 2).reshape(len(x), -1)
    ], axis=1)
    
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    predictions = matrix @ coefficients
    residuals = y.reshape(-1, 1) - predictions
    rmse = np.sqrt(np.mean(residuals**2))
    return coefficients, rmse

def predict_using_harmonic_model(x, coefficients, cosine_terms, sine_terms):
    """
    Uses the harmonic model coefficients to predict y values for a given set of x values.
    """
    x = x.reshape(-1, 1)
    matrix = np.concatenate([
        np.ones_like(x),
        np.array([np.cos(2 * np.pi * i * x) for i in range(1, cosine_terms + 1)]).transpose(1, 0, 2).reshape(len(x), -1),
        np.array([np.sin(2 * np.pi * i * x) for i in range(1, sine_terms + 1)]).transpose(1, 0, 2).reshape(len(x), -1)
    ], axis=1)
    
    predictions = matrix @ coefficients
    return predictions.flatten()

def compute_aic(y_actual, y_predicted, parameter_count):
    """
    Computes the Akaike Information Criterion (AIC) for the model.
    """
    residuals = y_actual - y_predicted
    sum_squared_residuals = np.sum(residuals**2)
    return 2 * parameter_count + len(y_actual) * np.log(sum_squared_residuals / len(y_actual))

def perform_k_fold_cross_validation(x_data, y_data, folds, cosine_terms, sine_terms):
    """
    Performs K-fold cross-validation on the harmonic model.
    """
    cross_validator = KFold(n_splits=folds, shuffle=True)
    aic_scores, rmse_scores = [], []
    
    for train_indices, test_indices in cross_validator.split(x_data):
        x_train, x_test = x_data[train_indices], x_data[test_indices]
        y_train, y_test = y_data[train_indices], y_data[test_indices]
        
        coefficients, _ = estimate_harmonic_model_coefficients(x_train, y_train, cosine_terms, sine_terms)
        y_predicted = predict_using_harmonic_model(x_test, coefficients, cosine_terms, sine_terms)
        
        rmse = np.sqrt(np.mean((y_test - y_predicted) ** 2))
        aic = compute_aic(y_test, y_predicted, len(coefficients))
        
        rmse_scores.append(rmse)
        aic_scores.append(aic)

    return np.mean(aic_scores), np.mean(rmse_scores), np.std(rmse_scores)

def execute_bootstrap_analysis(x_data, y_data, cosine_terms, sine_terms, iterations=100):
    """
    Performs bootstrapping to estimate model accuracy.
    """
    rmse_scores = []
    for _ in range(iterations):
        sample_indices = np.random.choice(len(x_data), size=len(x_data), replace=True)
        x_sample, y_sample = x_data[sample_indices], y_data[sample_indices]
        
        coefficients, _ = estimate_harmonic_model_coefficients(x_sample, y_sample, cosine_terms, sine_terms)
        y_predicted = predict_using_harmonic_model(x_data, coefficients, cosine_terms, sine_terms)
        
        rmse = np.sqrt(np.mean((y_data - y_predicted) ** 2))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores), np.std(rmse_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input_file", help="Name of input data file", default="sample.csv")
    args = parser.parse_args()

    # Set fixed cosine and sine terms for the model
    cosine_terms, sine_terms = 4, 4

    data = np.genfromtxt(args.input_file, delimiter=',', skip_header=1)
    x_values, y_values = data[:, 0], data[:, 1]

    # Model fitting and RMSE computation
    coefficients, model_rmse = estimate_harmonic_model_coefficients(x_values, y_values, cosine_terms, sine_terms)
    print(f"RMSE: {model_rmse}")

    y_predictions = predict_using_harmonic_model(x_values, coefficients, cosine_terms, sine_terms)
    aic_score = compute_aic(y_values.ravel(), y_predictions, len(coefficients))
    print(f"AIC: {aic_score}")

    # Perform cross-validation and bootstrapping
    aic_5_fold, rmse_5_fold, std_5_fold = perform_k_fold_cross_validation(x_values, y_values, 5, cosine_terms, sine_terms)
    print(f"5-Fold CV - AIC: {aic_5_fold}, RMSE: {rmse_5_fold}, Std Dev: {std_5_fold}")

    aic_10_fold, rmse_10_fold, std_10_fold = perform_k_fold_cross_validation(x_values, y_values, 10, cosine_terms, sine_terms)
    print(f"10-Fold CV - AIC: {aic_10_fold}, RMSE: {rmse_10_fold}, Std Dev: {std_10_fold}")

    mean_rmse_bootstrap, std_rmse_bootstrap = execute_bootstrap_analysis(x_values, y_values, cosine_terms, sine_terms)
    print(f"Bootstrap - Mean RMSE: {mean_rmse_bootstrap}, Std Dev: {std_rmse_bootstrap}")

if __name__ == "__main__":
    main()
    