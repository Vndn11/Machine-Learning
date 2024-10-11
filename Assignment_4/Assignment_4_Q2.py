import pandas as pd
import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
print("Version of Your Neural Network:",sklearn.__version__)
warnings.filterwarnings("ignore")

train_data = pd.read_csv('WineQuality_Train.csv')
test_data = pd.read_csv('WineQuality_Test.csv')

train_X = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
train_y = train_data['quality_grp']
test_X = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
test_y = test_data['quality_grp']

proportion_training = train_y.mean()
proportion_testing = test_y.mean()

print(f"Proportion of quality_grp = 1 in training partition: {proportion_training:.2f}")
print(f"Proportion of quality_grp = 1 in testing partition: {proportion_testing:.2f}")

grid_parameter = {
    'hidden_layer_sizes': [(layers,) * neurons for layers in range(1, 11) for neurons in range(2, 11, 2)],
    'activation': ['tanh', 'identity', 'relu']
}

list_results = []

for p in ParameterGrid(grid_parameter):
    
    start_time = time.time()
    
    mlp = MLPClassifier(max_iter=10000, random_state=2023484, **p)
    mlp.fit(train_X, train_y)
    
    y_pred = mlp.predict(test_X)
    rmse = np.sqrt(mean_squared_error(test_y, y_pred))
    misclassification_rate = 1 - accuracy_score(test_y, y_pred)
    
    list_results.append({
        'activation': p['activation'],
        'num_layers': len(p['hidden_layer_sizes']),
        'neurons_per_layer': p['hidden_layer_sizes'][0],
        'n_iter_': mlp.n_iter_,
        'best_loss_': mlp.best_loss_,
        'rmse': rmse,
        'misclassification_rate': misclassification_rate,
        'elapsed_time': time.time() - start_time
    })

df_results = pd.DataFrame(list_results)

df_results.sort_values(by=['rmse', 'misclassification_rate'], ascending=True, inplace=True)
df_results.to_csv("result_grid_search.csv")
print(df_results)

misclassification_sort_results = df_results.sort_values(by=['misclassification_rate','neurons_per_layer'])
results_sorted_by_rmse = df_results.sort_values(by=['rmse','neurons_per_layer'])

best_by_misclassification = misclassification_sort_results.iloc[0]
best_by_rmse = results_sorted_by_rmse.iloc[0]

print(f"Best network by misclassification rate:\n", best_by_misclassification)
print(f"Best network by RMSE:\n", best_by_rmse)

params_best = df_results.iloc[df_results['rmse'].argmin()]
best_mlp = MLPClassifier(
    hidden_layer_sizes=(params_best['num_layers'],) * params_best['neurons_per_layer'],
    activation=params_best['activation'],
    random_state=2023484,
    max_iter=10000
)

best_mlp.fit(train_X, train_y)

prob_y = best_mlp.predict_proba(test_X)[:, 1] 

threshold = 1.5 * proportion_training

adj_y_pred = np.where(prob_y >= threshold, 1, 0)

test_data['predicted_prob'] = prob_y
test_data['predicted_class'] = adj_y_pred

new_misclassification_rate = 1 - accuracy_score(test_y, adj_y_pred)

sns.boxplot(x='quality_grp', y='predicted_prob', data=test_data)
plt.axhline(y=threshold, color='r', linestyle='--')  
plt.title('Predicted Probability of quality_grp = 1 with Adjusted Threshold')
plt.xlabel('Observed quality_grp Category')
plt.ylabel('Predicted Probability')
plt.show()

print(f"New misclassification rate with adjusted threshold: {new_misclassification_rate:.4f}")
