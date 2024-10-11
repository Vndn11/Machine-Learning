import argparse
import csv
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# This function will convert the model coefficients to a string format.
def model_to_string(coeffs, m, n):
    # Check if coeffs[0] is a scalar or a single-element array, and format it accordingly
    first_coeff = coeffs[0] if isinstance(coeffs[0], np.float64) else coeffs[0][0]
    ones = f"{first_coeff:+.5f} "
    cosines = [f"{coeffs[j]:+.5f} cos(2π * {j})" for j in range(1, m+1)]
    sines = [f"{coeffs[j+m]:+.5f} sin(2π * {j})" for j in range(1, n+1)]
    return ones + " ".join(cosines) + " " + " ".join(sines)

# This function will fit the harmonic model and return the fitted coefficients and RMSE.
def fit_harmonic_model(x_sample, y_sample, m, n):
    x_sample = x_sample.reshape(-1, 1)
    y_sample = y_sample.ravel()  # Ensure y_sample is a flat, one-dimensional array
    ones = np.ones_like(x_sample)
    cosines = np.array([np.cos(2*np.pi*j*x_sample) for j in range(1, m+1)])[:,:,0].T
    sines = np.array([np.sin(2*np.pi*j*x_sample) for j in range(1, n+1)])[:,:,0].T
    dmatrix = np.concatenate([ones, cosines, sines], axis=1)
    
    # Solve using SVD for numerical stability
    coeffs = np.linalg.lstsq(dmatrix, y_sample, rcond=None)[0]
    # No need to reshape coeffs here, as it should be one-dimensional already
    model_stringified = model_to_string(coeffs, m, n)
    outputs = np.dot(dmatrix, coeffs)
    resids = y_sample.reshape(-1, 1) - outputs
    rmse = np.sqrt(np.mean(np.square(resids)))
    return coeffs, rmse

# Function to predict values using the harmonic model
def predict_harmonic_model(x_sample, coeffs, m, n):
    x_sample = x_sample.reshape(-1, 1)
    ones = np.ones_like(x_sample)
    cosines = np.array([np.cos(2*np.pi*j*x_sample) for j in range(1, m+1)])[:,:,0].T
    sines = np.array([np.sin(2*np.pi*j*x_sample) for j in range(1, n+1)])[:,:,0].T
    dmatrix = np.concatenate([ones, cosines, sines], axis=1)
    
    y_pred = np.dot(dmatrix, coeffs)
    return y_pred.flatten()

# This function calculates AIC.
def calculate_aic(y, y_pred, k):
    resid = y - y_pred
    rss = np.sum(resid**2)
    aic = 2*k + len(y) * np.log(rss/len(y))
    return aic

# This function performs K-fold cross-validation.
def k_fold_cv(data_x, data_y, k, m, n):
    kf = KFold(n_splits=k, shuffle=True)
    aic_values = []
    rmse_values = []

    for train_index, test_index in kf.split(data_x):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]
        
        coeffs, _ = fit_harmonic_model(x_train, y_train, m, n)
        y_pred = predict_harmonic_model(x_test, coeffs, m, n)
        
        # Calculate RMSE and AIC for this fold
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        aic = calculate_aic(y_test, y_pred, len(coeffs))
        
        rmse_values.append(rmse)
        aic_values.append(aic)

    return np.mean(aic_values), np.mean(rmse_values), np.std(rmse_values)

# This function performs the bootstrapping.
def bootstrap(data_x, data_y, m, n, num_bootstrap=100):
    rmse_values = []
    for _ in range(num_bootstrap):
        indices = np.random.choice(range(len(data_x)), size=len(data_x), replace=True)
        x_bootstrap, y_bootstrap = data_x[indices], data_y[indices]
        coeffs, _ = fit_harmonic_model(x_bootstrap, y_bootstrap, m, n)
        y_pred = predict_harmonic_model(data_x, coeffs, m, n)
    
        rmse = np.sqrt(np.mean((data_y - y_pred) ** 2))
        rmse_values.append(rmse)

    return np.mean(rmse_values), np.std(rmse_values)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("m", help="Number of cosine terms", type=int, default="3")
    # parser.add_argument("n", help="Number of sine terms", type=int, default="3")
    parser.add_argument("-f", "--input_file", help="Name of input data file", default="sample.csv")
    args = parser.parse_args()

    args.m = 4
    args.n = 4

    data = np.genfromtxt(args.input_file, delimiter=',', skip_header=1)
    x_sample = data[:, 0]
    y_sample = data[:, 1]

    # Fit the model and calculate RMSE
    coeffs, rmse = fit_harmonic_model(x_sample, y_sample, args.m, args.n)

    # Print the string representation of the model and RMSE
    model_str = model_to_string(coeffs, args.m, args.n)
    print(f"RMSE: {rmse}")

    y_pred = predict_harmonic_model(x_sample, coeffs, args.m, args.n)
    
    # plt.scatter(x_sample, y_sample, color='blue', alpha=0.5, label='Original data')

    # # Sort x_sample and corresponding y_pred for plotting
    # sorted_indices = np.argsort(x_sample)
    # sorted_x = x_sample[sorted_indices]
    # sorted_y_pred = y_pred[sorted_indices]

    # # Plot model predictions
    # plt.plot(sorted_x, sorted_y_pred, color='red', label='Model prediction')

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Harmonic Model Fit')
    # plt.legend()
    # plt.show()
    
    aic = calculate_aic(y_sample.ravel(), y_pred, len(coeffs))
    print(f"AIC: {aic}")

    # Perform K-fold CV
    aic_k5, rmse_k5, std_k5 = k_fold_cv(x_sample, y_sample, 5, args.m, args.n)
    print(f"5-Fold CV - AIC: {aic_k5}, RMSE: {rmse_k5}, Std Dev: {std_k5}")

    aic_k10, rmse_k10, std_k10 = k_fold_cv(x_sample, y_sample, 10, args.m, args.n)
    print(f"10-Fold CV - AIC: {aic_k10}, RMSE: {rmse_k10}, Std Dev: {std_k10}")

    # Perform Bootstrapping
    mean_rmse_bs, std_rmse_bs = bootstrap(x_sample, y_sample, args.m, args.n)
    print(f"Bootstrap - Mean RMSE: {mean_rmse_bs}, Std Dev: {std_rmse_bs}")

if __name__=="__main__":
    main()